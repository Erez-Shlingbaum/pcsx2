// SPDX-FileCopyrightText: 2002-2025 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/HW/GSTextureUpscaler.h"

#include "Config.h"
#include "GS/GSExtra.h"
#include "GS/Renderers/HW/GSTextureReplacements.h"
#include "Host.h"
#include "IconsFontAwesome6.h"
#include "VMManager.h"

#include "common/Console.h"
#include "common/FileSystem.h"
#include "common/Path.h"
#include "common/StringUtil.h"
#include "common/Threading.h"

#include "fmt/format.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include "common/RedtapeWindows.h"
#include <wil/resource.h>
#else
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace GSTextureUpscaler
{
	struct Job
	{
		std::string input;
		std::string output;
		// Config snapshot captured at enqueue time, so the worker never races UpdateConfig().
		std::string exe_path;
		std::string model_dir;
		std::string model_name;
		int scale;
		int passes;
		int gpu_id;
		int tile_size;
	};

	class ExternalProcess;

	// Hard ceiling on chained passes, independent of whatever a hand-edited ini says - see
	// TextureUpscalerPasses in Config.h. 3 passes already means scale^3 (e.g. 4x -> 64x), which is
	// already an extreme setting; this just bounds how far a bad config value can run.
	static constexpr int MAX_UPSCALER_PASSES = 3;

	// Cached config snapshot (read on the GS thread via UpdateConfig, consumed by the worker).
	// s_enabled is atomic so IsEnabled() never contends on s_mutex from the GS thread.
	static std::atomic<bool> s_enabled{false};
	static std::string s_exe_path;
	static std::string s_model_dir;
	static std::string s_model_name;
	static int s_scale = 4;
	static int s_passes = 1;
	static int s_gpu_id = -1;
	static int s_tile_size = 0;

	static std::mutex s_mutex;
	static std::condition_variable s_cv;
	static std::thread s_worker_thread;
	static std::deque<Job> s_queue;
	// Pending directory catch-up scans (dumps_dir, replacements_dir), run on the worker thread so
	// their slow filesystem enumeration and per-file header reads never touch the GS thread.
	static std::deque<std::pair<std::string, std::string>> s_scan_requests;
	// Pending per-dump upscale requests (dump_path, replacement_path) enqueued by the texture-dump
	// work item. Their filesystem pre-checks (min-size gate + exists check) are run here on the
	// worker thread rather than on the GSTextureReplacements worker the GS thread waits on.
	static std::deque<std::pair<std::string, std::string>> s_pending_checks;
	// Output paths that are queued or in flight, to avoid duplicate work.
	static std::unordered_set<std::string> s_pending_outputs;
	static bool s_worker_running = false;
	// Atomic because RunUpscalerBatch (called with s_mutex released, so a long-running external
	// process doesn't block the whole queue) checks this between passes to bail out promptly.
	static std::atomic<bool> s_stop{false};

	// Auto-apply signalling (read/written across threads without the main lock).
	static std::atomic<bool> s_reload_requested{false};
	static std::atomic<u32> s_min_size{32};

	// Progress reporting for the OSD (guarded by s_mutex). done counts processed jobs (so the
	// running "x/y" always reaches y), ok counts only successes (what the completion toast shows).
	static u32 s_progress_total = 0;
	static u32 s_progress_done = 0;
	static u32 s_progress_ok = 0;

	// Successfully written replacement files awaiting registration on the GS thread (s_mutex).
	static std::vector<std::string> s_completed_files;

	// Largest number of textures handed to a single upscaler process in batch mode.
	static constexpr size_t MAX_BATCH_SIZE = 512;

	// When work trickles in one texture at a time (e.g. exploring a new area), wait briefly for
	// more to arrive so several textures share one process/model-load, instead of paying the
	// model load for every single texture.
	static constexpr std::chrono::milliseconds COALESCE_WAIT{2500};
	static constexpr size_t COALESCE_TARGET = 8;

	// The process currently being waited on, so Shutdown() can terminate it from another thread.
	// Points at a stack-local in RunProcessAndWait; set/cleared under s_mutex.
	static ExternalProcess* s_current_process = nullptr;

	// Original alpha plane of a dump, kept aside while the model upscales the color data. The model
	// must never see real PS2 alpha: PS2 uses 0x80 as fully-opaque, but AI upscalers treat 128 as
	// half-transparent and premultiply the color against black, mangling it. So the model only ever
	// sees a forced-opaque copy, and afterwards we restore the original's exact (resampled) alpha.
	// Only the alpha plane is retained; the model's color passes through untouched.
	struct SourceImage
	{
		u32 width = 0;
		u32 height = 0;
		std::vector<u8> alpha; // tight, width*1 pitch
		bool valid = false;
	};

	// RAII wrapper around a spawned child process. Encapsulates the CreateProcessW vs posix_spawnp
	// difference so RunProcessAndWait has a single, platform-agnostic code path, and owns the
	// handle/pid so it's released on scope exit. Terminate() is safe to call from another thread
	// while Wait() blocks (Shutdown() uses this to kill a long-running upscale).
	class ExternalProcess
	{
	public:
		ExternalProcess() = default;
		ExternalProcess(const ExternalProcess&) = delete;
		ExternalProcess& operator=(const ExternalProcess&) = delete;
#ifdef _WIN32
		ExternalProcess(ExternalProcess&&) = default;
		ExternalProcess& operator=(ExternalProcess&&) = default;
#else
		ExternalProcess(ExternalProcess&& o) noexcept : m_pid(o.m_pid) { o.m_pid = -1; }
		ExternalProcess& operator=(ExternalProcess&& o) noexcept
		{
			std::swap(m_pid, o.m_pid);
			return *this;
		}

		// Reaps the child. Wait() deliberately leaves it a zombie (WNOWAIT) so the pid can't be
		// recycled while Shutdown() may still Terminate() us through s_current_process; reaping
		// here, after RunProcessAndWait has cleared that pointer, closes the race. Blocks only
		// if the process were still running, which RunProcessAndWait's unconditional Wait()
		// rules out.
		~ExternalProcess()
		{
			if (m_pid > 0)
			{
				int status = 0;
				while (waitpid(m_pid, &status, 0) < 0 && errno == EINTR)
				{
				}
			}
		}
#endif

		// Launches exe with args (each an unquoted token). Returns an invalid process on failure.
		static ExternalProcess Spawn(const std::string& exe, const std::vector<std::string>& args);

		bool IsValid() const
		{
#ifdef _WIN32
			return static_cast<bool>(m_process);
#else
			return m_pid > 0;
#endif
		}

		// Blocks until the process exits; returns true if it exited with code 0.
		bool Wait()
		{
#ifdef _WIN32
			WaitForSingleObject(m_process.get(), INFINITE);
			DWORD exit_code = 1;
			GetExitCodeProcess(m_process.get(), &exit_code);
			return (exit_code == 0);
#else
			// WNOWAIT: observe the exit without reaping, so m_pid stays valid (as a zombie)
			// until the destructor reaps it - otherwise Terminate() from another thread could
			// kill() a recycled pid in the window before s_current_process is cleared.
			siginfo_t info = {};
			while (waitid(P_PID, static_cast<id_t>(m_pid), &info, WEXITED | WNOWAIT) < 0 && errno == EINTR)
			{
			}
			return (info.si_code == CLD_EXITED && info.si_status == 0);
#endif
		}

		// Force-kills the process. Safe to call from another thread while Wait() is in flight.
		void Terminate()
		{
#ifdef _WIN32
			if (m_process)
				TerminateProcess(m_process.get(), 1);
#else
			if (m_pid > 0)
				kill(m_pid, SIGKILL);
#endif
		}

	private:
#ifdef _WIN32
		static std::wstring QuoteArg(std::string_view arg);
		wil::unique_handle m_process;
		wil::unique_handle m_thread;
#else
		pid_t m_pid = -1;
#endif
	};

	static void StartWorkerLocked();
	static void WorkerThreadEntryPoint();
	static bool RunProcessAndWait(const std::string& exe, const std::vector<std::string>& args);
	static bool StageOpaqueInput(const std::string& src_path, const std::string& staged_path);
	static SourceImage DecodeSource(const std::string& path);
	static bool PostprocessAndSave(const std::string& model_output_path, const SourceImage& source, const std::string& save_path);
	static u32 RunUpscalerBatch(const std::vector<Job>& jobs, std::vector<std::string>* completed);
	static void ScanDirectoryAndEnqueue(const std::string& dumps_dir, const std::string& replacements_dir);
	static void UpdateProgressOSD(bool finished);
	static std::string AutoDetectUpscalerExe();
	static std::string DeriveUpscalerModelDir(const std::string& exe);
	static std::string DefaultUpscalerModelName(const std::string& model_dir);
} // namespace GSTextureUpscaler

// Probes well-known Upscayl desktop install locations. Returns the CLI backend path if found.
std::string GSTextureUpscaler::AutoDetectUpscalerExe()
{
	const auto join = [](std::string base, std::initializer_list<const char*> parts) {
		for (const char* p : parts)
			base = Path::Combine(base, p);
		return base;
	};

	std::vector<std::string> candidates;
#ifdef _WIN32
	for (const char* env : {"ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"})
	{
		if (const char* base = std::getenv(env))
			candidates.push_back(join(base, {"Upscayl", "resources", "bin", "upscayl-bin.exe"}));
	}
	if (const char* lad = std::getenv("LOCALAPPDATA"))
		candidates.push_back(join(lad, {"Programs", "Upscayl", "resources", "bin", "upscayl-bin.exe"}));
#elif defined(__APPLE__)
	candidates.push_back("/Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin");
	if (const char* home = std::getenv("HOME"))
		candidates.push_back(join(home, {"Applications", "Upscayl.app", "Contents", "Resources", "bin", "upscayl-bin"}));
#else
	candidates.push_back("/opt/Upscayl/resources/bin/upscayl-bin");
	candidates.push_back("/usr/lib/upscayl/resources/bin/upscayl-bin");
	candidates.push_back("/usr/share/upscayl/resources/bin/upscayl-bin");
	// Flatpak
	if (const char* home = std::getenv("HOME"))
		candidates.push_back(join(home, {".var", "app", "org.upscayl.Upscayl", "resources", "bin", "upscayl-bin"}));
#endif

	for (const std::string& c : candidates)
	{
		if (FileSystem::FileExists(c.c_str()))
			return c;
	}
	return std::string();
}

// Given .../resources/bin/upscayl-bin[.exe], the models live in .../resources/models.
std::string GSTextureUpscaler::DeriveUpscalerModelDir(const std::string& exe)
{
	if (exe.empty())
		return std::string();

	const std::string bin_dir(Path::GetDirectory(exe));
	const std::string resources_dir(Path::GetDirectory(bin_dir));
	std::string models(Path::Combine(resources_dir, "models"));
	if (FileSystem::DirectoryExists(models.c_str()))
		return models;
	return std::string();
}

// Upscayl renames the bundled models, so the CLI's own default (realesrgan-x4plus) isn't present.
// Prefer a well-known general-purpose model, but fall back to whatever .param the install actually
// ships - a hardcoded name list rots as Upscayl rotates its bundled models, and coming up empty
// makes the CLI fail later with a cryptic "model not found".
std::string GSTextureUpscaler::DefaultUpscalerModelName(const std::string& model_dir)
{
	if (model_dir.empty())
		return std::string();

	for (const char* name : {"upscayl-standard-4x", "ultrasharp-4x", "realesrgan-x4plus"})
	{
		if (FileSystem::FileExists(Path::Combine(model_dir, fmt::format("{}.param", name)).c_str()))
			return name;
	}

	// No preferred model present - use any .param in the directory.
	FileSystem::FindResultsArray files;
	if (FileSystem::FindFiles(model_dir.c_str(), "*.param", FILESYSTEM_FIND_FILES, &files) && !files.empty())
		return std::string(Path::GetFileTitle(files.front().FileName));

	return std::string();
}

void GSTextureUpscaler::UpdateConfig()
{
	const Pcsx2Config::GSOptions& gs = GSConfig;
	const bool want_enabled = gs.UpscaleReplacementTextures;

	// Resolve blank fields by auto-detecting an installed Upscayl. Do the filesystem probing
	// BEFORE taking the lock (so the worker's brief lock acquisitions aren't blocked on disk I/O),
	// and skip it entirely when the feature is off - there's nothing to detect for.
	std::string exe, model_dir, model_name;
	if (want_enabled)
	{
		exe = gs.TextureUpscalerPath;
		if (exe.empty())
			exe = AutoDetectUpscalerExe();

		model_dir = gs.TextureUpscalerModelDir;
		if (model_dir.empty())
			model_dir = DeriveUpscalerModelDir(exe);

		model_name = gs.TextureUpscalerModelName;
		if (model_name.empty())
			model_name = DefaultUpscalerModelName(model_dir);
	}

	std::unique_lock<std::mutex> lock(s_mutex);
	s_exe_path = std::move(exe);
	s_model_dir = std::move(model_dir);
	s_model_name = std::move(model_name);
	s_scale = gs.TextureUpscalerScale;
	s_passes = std::clamp(gs.TextureUpscalerPasses, 1, MAX_UPSCALER_PASSES);
	s_gpu_id = gs.TextureUpscalerGpuId;
	s_tile_size = gs.TextureUpscalerTileSize;
	s_enabled.store(want_enabled && !s_exe_path.empty(), std::memory_order_relaxed);

	s_min_size.store(static_cast<u32>(std::max(gs.TextureUpscalerMinSize, 0)), std::memory_order_relaxed);
}

bool GSTextureUpscaler::TakeCompletedFiles(std::vector<std::string>* files)
{
	// Hand back at most this many per frame; registering a file queues a texture load + GPU
	// upload, and doing hundreds in one vsync (e.g. when a big batch finishes) causes a long
	// stall. Spreading them over frames keeps the game responsive.
	static constexpr size_t MAX_FILES_PER_CALL = 8;

	// The atomic is the cheap per-frame fast path; only take the lock when there's work.
	if (!s_reload_requested.load(std::memory_order_relaxed))
		return false;

	std::unique_lock<std::mutex> lock(s_mutex);
	if (s_completed_files.empty())
	{
		s_reload_requested.store(false, std::memory_order_relaxed);
		return false;
	}

	const size_t n = std::min(s_completed_files.size(), MAX_FILES_PER_CALL);
	files->assign(std::make_move_iterator(s_completed_files.begin()), std::make_move_iterator(s_completed_files.begin() + n));
	s_completed_files.erase(s_completed_files.begin(), s_completed_files.begin() + n);
	if (s_completed_files.empty())
		s_reload_requested.store(false, std::memory_order_relaxed);
	return true;
}

bool GSTextureUpscaler::IsEnabled()
{
	return s_enabled.load(std::memory_order_relaxed);
}

// Reads just the IHDR dimensions from a PNG without decoding it.
static bool GetPNGDimensions(const std::string& path, u32* width, u32* height)
{
	const auto fp = FileSystem::OpenManagedCFile(path.c_str(), "rb");
	u8 hdr[24];
	if (!fp || std::fread(hdr, 1, sizeof(hdr), fp.get()) != sizeof(hdr) ||
		std::memcmp(hdr, "\x89PNG\r\n\x1a\n", 8) != 0 || std::memcmp(hdr + 12, "IHDR", 4) != 0)
	{
		return false;
	}

	*width = (static_cast<u32>(hdr[16]) << 24) | (static_cast<u32>(hdr[17]) << 16) | (static_cast<u32>(hdr[18]) << 8) | hdr[19];
	*height = (static_cast<u32>(hdr[20]) << 24) | (static_cast<u32>(hdr[21]) << 16) | (static_cast<u32>(hdr[22]) << 8) | hdr[23];
	return true;
}

void GSTextureUpscaler::QueueUpscale(std::string dump_path, std::string replacement_path, bool skip_exists_check)
{
	if (!IsEnabled())
		return;

	// AI models produce garbage on tiny inputs (icons, dithered fills, gradient strips) - they
	// amplify the dithering into noise. Leave textures below the threshold at native size.
	const u32 min_size = s_min_size.load(std::memory_order_relaxed);
	if (min_size > 0)
	{
		u32 width, height;
		if (GetPNGDimensions(dump_path, &width, &height) && std::min(width, height) < min_size)
			return;
	}

	// Check the filesystem BEFORE taking the lock; this can be a slow disk (or cloud-synced
	// folder) round-trip, and the GS thread contends on this mutex.
	if (!skip_exists_check && FileSystem::FileExists(replacement_path.c_str()))
		return;

	std::unique_lock<std::mutex> lock(s_mutex);
	if (!s_enabled.load(std::memory_order_relaxed))
		return;

	// Don't redo work that's already queued/running.
	if (s_pending_outputs.find(replacement_path) != s_pending_outputs.end())
		return;

	s_pending_outputs.insert(replacement_path);
	s_queue.push_back(Job{std::move(dump_path), std::move(replacement_path), s_exe_path, s_model_dir,
		s_model_name, s_scale, s_passes, s_gpu_id, s_tile_size});
	s_progress_total++;

	StartWorkerLocked();
	s_cv.notify_one();
}

void GSTextureUpscaler::QueueUpscaleFromDump(std::string dump_path, std::string replacement_path)
{
	if (!IsEnabled())
		return;

	// Deliberately does NO filesystem work here - just hands the paths to the worker under the lock.
	// This runs on the GSTextureReplacements worker thread, which the GS thread busy-waits on via
	// SyncWorkerThread(); the min-size gate (re-reads the dump PNG) and the replacement-exists check
	// (stats the replacements dir - the very folder that may be cloud-synced and slow) are deferred
	// to our own worker, which picks the request up and calls QueueUpscale() off the GS-thread path.
	std::unique_lock<std::mutex> lock(s_mutex);
	if (!s_enabled.load(std::memory_order_relaxed))
		return;

	s_pending_checks.emplace_back(std::move(dump_path), std::move(replacement_path));
	StartWorkerLocked();
	s_cv.notify_one();
}

void GSTextureUpscaler::QueueDirectoryScan(std::string dumps_dir, std::string replacements_dir)
{
	if (!IsEnabled())
		return;

	std::unique_lock<std::mutex> lock(s_mutex);
	if (!s_enabled.load(std::memory_order_relaxed))
		return;

	s_scan_requests.emplace_back(std::move(dumps_dir), std::move(replacements_dir));
	StartWorkerLocked();
	s_cv.notify_one();
}

void GSTextureUpscaler::StartWorkerLocked()
{
	if (s_worker_running)
		return;

	s_stop.store(false, std::memory_order_relaxed);
	s_worker_running = true;
	s_worker_thread = std::thread(WorkerThreadEntryPoint);
}

void GSTextureUpscaler::CancelPendingJobs()
{
	std::unique_lock<std::mutex> lock(s_mutex);
	s_queue.clear();
	s_scan_requests.clear();
	s_pending_checks.clear();
	s_pending_outputs.clear();
	s_completed_files.clear();
	s_reload_requested.store(false, std::memory_order_relaxed);
	s_progress_total = 0;
	s_progress_done = 0;
	s_progress_ok = 0;
	// Clear immediately rather than letting a stale "N/M" (from the game just left) sit for up to
	// its 60-second refresh window.
	Host::RemoveKeyedOSDMessage("TextureUpscaleProgress");

	// A process already in flight is left to finish on its own rather than force-killed - it's
	// nearly done either way, and killing it would need the same Terminate()/Wait() dance as
	// Shutdown() without the actual benefit of stopping the worker thread. Its completion will
	// still register normally; s_pending_outputs.erase() on an already-cleared set is a no-op.
}

void GSTextureUpscaler::Shutdown()
{
	std::thread worker_to_join;
	{
		std::unique_lock<std::mutex> lock(s_mutex);
		s_stop.store(true, std::memory_order_relaxed);
		// Clear enabled under the lock so any QueueUpscale() racing in after shutdown (e.g. from a
		// texture-dump work item still draining on the replacements worker) is rejected by its
		// under-lock re-check instead of resurrecting the worker onto the joined static thread.
		s_enabled.store(false, std::memory_order_relaxed);
		s_queue.clear();
		s_scan_requests.clear();
		s_pending_checks.clear();
		s_pending_outputs.clear();
		s_completed_files.clear();
		s_reload_requested.store(false, std::memory_order_relaxed);
		s_progress_total = 0;
		s_progress_done = 0;
		s_progress_ok = 0;

		// Terminate an in-flight process so shutdown doesn't block on a long upscale.
		if (s_current_process)
			s_current_process->Terminate();

		s_cv.notify_all();
		worker_to_join = std::move(s_worker_thread);
	}

	if (worker_to_join.joinable())
		worker_to_join.join();
}

void GSTextureUpscaler::UpdateProgressOSD(bool finished)
{
	// Called with s_mutex held. AddKeyedOSDMessage is thread-safe.
	if (finished)
	{
		// Report successes only; failures were already warned about on the console.
		Host::AddIconOSDMessage("TextureUpscaleProgress", ICON_FA_WAND_MAGIC_SPARKLES,
			fmt::format(TRANSLATE_FS("TextureReplacement", "AI texture upscaling complete ({} textures)."), s_progress_ok),
			Host::OSD_INFO_DURATION);
		s_progress_total = 0;
		s_progress_done = 0;
		s_progress_ok = 0;
	}
	else
	{
		Host::AddIconOSDMessage("TextureUpscaleProgress", ICON_FA_WAND_MAGIC_SPARKLES,
			fmt::format(TRANSLATE_FS("TextureReplacement", "AI upscaling textures... {}/{}"), s_progress_done, s_progress_total),
			60.0f);
	}
}

// Defined with the other staging-directory helpers below.
static void CleanupStaleWorkDirectories();

void GSTextureUpscaler::WorkerThreadEntryPoint()
{
	Threading::SetNameOfCurrentThread("Texture Upscaler");

	// Sweep staging directories orphaned by crashed/killed instances before doing any work of
	// our own. Cheap when there's nothing stale, and we're already off the GS thread here.
	CleanupStaleWorkDirectories();

	std::unique_lock<std::mutex> lock(s_mutex);
	for (;;)
	{
		s_cv.wait(lock, []() {
			return s_stop.load(std::memory_order_relaxed) || !s_queue.empty() || !s_scan_requests.empty() ||
				   !s_pending_checks.empty();
		});
		if (s_stop.load(std::memory_order_relaxed))
			break;

		// Handle a pending catch-up scan (slow directory enumeration + per-file header reads) here
		// on the worker, off the GS thread. It enqueues jobs via QueueUpscale(), so run it with the
		// lock released, then loop back to pick up whatever it queued.
		if (!s_scan_requests.empty())
		{
			const std::pair<std::string, std::string> req(std::move(s_scan_requests.front()));
			s_scan_requests.pop_front();
			lock.unlock();
			ScanDirectoryAndEnqueue(req.first, req.second);
			lock.lock();
			continue;
		}

		// Handle a deferred per-dump upscale request. QueueUpscale() does the min-size gate and the
		// exists check (both filesystem round-trips) here on the worker, off the GS-thread path -
		// see QueueUpscaleFromDump(). Run it with the lock released (it re-takes s_mutex to enqueue
		// the job), then loop back so the job it queued gets coalesced into the batch below.
		if (!s_pending_checks.empty())
		{
			const std::pair<std::string, std::string> req(std::move(s_pending_checks.front()));
			s_pending_checks.pop_front();
			lock.unlock();
			QueueUpscale(req.first, req.second);
			lock.lock();
			continue;
		}

		// If only a few textures are queued, linger a moment - during gameplay they trickle in
		// as new areas load, and coalescing them into one process avoids a model load each.
		if (s_queue.size() < COALESCE_TARGET)
			s_cv.wait_for(lock, COALESCE_WAIT, []() { return s_stop.load(std::memory_order_relaxed) || s_queue.size() >= COALESCE_TARGET; });
		if (s_stop.load(std::memory_order_relaxed))
			break;

		// Drain a batch of jobs which share the same upscaler configuration. Handing the whole
		// batch to one process means the AI model is loaded once instead of per-texture, which
		// is where nearly all of the per-file wall time goes.
		std::vector<Job> batch;
		batch.push_back(std::move(s_queue.front()));
		s_queue.pop_front();
		while (!s_queue.empty() && batch.size() < MAX_BATCH_SIZE)
		{
			const Job& next = s_queue.front();
			const Job& first = batch.front();
			if (next.exe_path != first.exe_path || next.model_dir != first.model_dir ||
				next.model_name != first.model_name || next.scale != first.scale ||
				next.passes != first.passes || next.gpu_id != first.gpu_id || next.tile_size != first.tile_size)
			{
				break;
			}
			batch.push_back(std::move(s_queue.front()));
			s_queue.pop_front();
		}

		UpdateProgressOSD(false);

		// Run the (blocking) external process without holding the lock.
		lock.unlock();
		std::vector<std::string> completed;
		const u32 ok_count = RunUpscalerBatch(batch, &completed);
		lock.lock();

		for (const Job& job : batch)
			s_pending_outputs.erase(job.output);

		// Shutdown() already cleared the progress counters and completed-file buffer; don't
		// repopulate them here or they leak into the next session. Dropping the completed list
		// is safe - the files are on disk, so the next ReloadReplacementMap() picks them up.
		if (!s_stop.load(std::memory_order_relaxed))
		{
			s_progress_done += static_cast<u32>(batch.size());
			s_progress_ok += ok_count;

			if (!completed.empty())
			{
				// Hand the finished files to the GS thread, which registers and applies them
				// (debounced) via TakeCompletedFiles().
				s_completed_files.insert(s_completed_files.end(), std::make_move_iterator(completed.begin()),
					std::make_move_iterator(completed.end()));
				s_reload_requested.store(true, std::memory_order_relaxed);
			}
			if (ok_count < batch.size())
				Console.WarningFmt("Texture upscaling failed for {} of {} texture(s).", batch.size() - ok_count, batch.size());

			if (s_queue.empty())
				UpdateProgressOSD(true);
		}
	}

	s_worker_running = false;
}

// Builds the argument list common to Real-ESRGAN-ncnn-vulkan / upscayl-bin CLIs. Input/output
// may be files (single mode) or directories (batch mode - the CLI natively supports both).
// Kept as separate string tokens; platform code below handles quoting/escaping.
static std::vector<std::string> BuildUpscalerArgs(
	const GSTextureUpscaler::Job& job, const std::string& input, const std::string& output)
{
	std::vector<std::string> args;
	args.push_back("-i");
	args.push_back(input);
	args.push_back("-o");
	args.push_back(output);
	args.push_back("-f");
	args.push_back("png");
	if (job.scale > 0)
	{
		args.push_back("-s");
		args.push_back(std::to_string(job.scale));
	}
	if (!job.model_dir.empty())
	{
		args.push_back("-m");
		args.push_back(job.model_dir);
	}
	if (!job.model_name.empty())
	{
		args.push_back("-n");
		args.push_back(job.model_name);
	}
	if (job.gpu_id >= 0)
	{
		args.push_back("-g");
		args.push_back(std::to_string(job.gpu_id));
	}
	// Tile size bounds how long each GPU compute dispatch runs. The CLI's default (0 = auto)
	// picks tiles as large as VRAM allows, and those long dispatches starve the game's 3D queue
	// - the emulator stalls with its GPU "utilization" (3D engine) dropping while the compute
	// engine is busy. While the game is actively running, a modest tile keeps dispatches short
	// so rendering interleaves smoothly; when it isn't (paused, menus, VM stopped), the GPU is
	// free and the upscaler can take tiles as large as it likes.
	int tile_size = job.tile_size;
	if (tile_size <= 0)
		tile_size = (VMManager::GetState() == VMState::Running) ? 64 : 0;
	if (tile_size > 0)
	{
		args.push_back("-t");
		args.push_back(std::to_string(tile_size));
	}
	return args;
}

#ifdef _WIN32

std::wstring GSTextureUpscaler::ExternalProcess::QuoteArg(std::string_view arg)
{
	// Standard CommandLineToArgvW quoting rules.
	std::wstring warg = StringUtil::UTF8StringToWideString(arg);
	if (!warg.empty() && warg.find_first_of(L" \t\"") == std::wstring::npos)
		return warg;

	std::wstring out;
	out.push_back(L'"');
	for (size_t i = 0; i < warg.size(); i++)
	{
		size_t backslashes = 0;
		while (i < warg.size() && warg[i] == L'\\')
		{
			i++;
			backslashes++;
		}

		if (i == warg.size())
		{
			out.append(backslashes * 2, L'\\');
			break;
		}
		else if (warg[i] == L'"')
		{
			out.append(backslashes * 2 + 1, L'\\');
			out.push_back(L'"');
		}
		else
		{
			out.append(backslashes, L'\\');
			out.push_back(warg[i]);
		}
	}
	out.push_back(L'"');
	return out;
}

GSTextureUpscaler::ExternalProcess GSTextureUpscaler::ExternalProcess::Spawn(
	const std::string& exe, const std::vector<std::string>& args)
{
	ExternalProcess proc;

	std::wstring cmdline = QuoteArg(exe);
	for (const std::string& arg : args)
	{
		cmdline.push_back(L' ');
		cmdline.append(QuoteArg(arg));
	}

	STARTUPINFOW si = {};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi = {};

	// CreateProcessW requires a mutable command-line buffer.
	std::vector<wchar_t> cmdbuf(cmdline.begin(), cmdline.end());
	cmdbuf.push_back(0);

	// Below-normal priority keeps the upscaler from starving the emulator while the game runs.
	if (!CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, FALSE,
			CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS, nullptr, nullptr, &si, &pi))
	{
		Console.ErrorFmt("Failed to launch texture upscaler '{}' (error {}).", exe,
			static_cast<u32>(GetLastError()));
		return proc;
	}

	proc.m_process.reset(pi.hProcess);
	proc.m_thread.reset(pi.hThread);
	return proc;
}

#else // POSIX

GSTextureUpscaler::ExternalProcess GSTextureUpscaler::ExternalProcess::Spawn(
	const std::string& exe, const std::vector<std::string>& args)
{
	ExternalProcess proc;

	std::vector<std::string> arg_strings;
	arg_strings.push_back(exe);
	for (const std::string& arg : args)
		arg_strings.push_back(arg);

	std::vector<char*> argv;
	argv.reserve(arg_strings.size() + 1);
	for (std::string& arg : arg_strings)
		argv.push_back(arg.data());
	argv.push_back(nullptr);

	pid_t pid = -1;
	const int spawn_res = posix_spawnp(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ);
	if (spawn_res != 0)
	{
		Console.ErrorFmt("Failed to launch texture upscaler '{}' (error {}).", exe, spawn_res);
		return proc;
	}

	proc.m_pid = pid;
	return proc;
}

#endif

bool GSTextureUpscaler::RunProcessAndWait(const std::string& exe, const std::vector<std::string>& args)
{
	ExternalProcess proc = ExternalProcess::Spawn(exe, args);
	if (!proc.IsValid())
		return false;

	{
		std::unique_lock<std::mutex> lock(s_mutex);
		if (s_stop.load(std::memory_order_relaxed))
		{
			// Shutdown() ran between the spawn and here, so it couldn't have terminated this
			// process; kill it ourselves rather than blocking Shutdown's join on a long upscale.
			lock.unlock();
			proc.Terminate();
			proc.Wait();
			return false;
		}
		s_current_process = &proc;
	}

	const bool ok = proc.Wait();

	{
		std::unique_lock<std::mutex> lock(s_mutex);
		s_current_process = nullptr;
	}

	return ok;
}

// Writes a fully-opaque copy of a dump for the model to consume (see SourceImage for why the model
// must never see PS2 alpha). Returns false if the dump couldn't be decoded/saved, in which case
// the caller stages the raw file instead. Deliberately keeps no pixel data resident - the source
// is re-decoded per file during the harvest pass (DecodeSource) so a whole batch's originals aren't
// pinned in memory at once.
bool GSTextureUpscaler::StageOpaqueInput(const std::string& src_path, const std::string& staged_path)
{
	const GSTextureReplacements::ReplacementTextureLoader loader = GSTextureReplacements::GetLoader(src_path);
	GSTextureReplacements::ReplacementTexture tex;
	if (!loader || !loader(src_path, &tex, true) || tex.format != GSTexture::Format::Color)
		return false;

	for (u32 y = 0; y < tex.height; y++)
	{
		u8* row = tex.data.data() + (static_cast<size_t>(y) * tex.pitch);
		for (u32 x = 0; x < tex.width; x++)
			row[(x * 4) + 3] = 0xFF;
	}

	return GSTextureReplacements::SavePNGImage(staged_path, tex.width, tex.height, tex.data.data(), tex.pitch);
}

// Decodes a dump's alpha plane, preserving its original (PS2) values. Used during the harvest
// pass to restore the source's exact alpha over the model's output, and as a pure decode-validity
// check for raw-staged files (see RunUpscalerBatch). Only the alpha plane is kept resident; the
// color channels are never needed again, so extracting the plane quarters the memory and the
// resample work below.
GSTextureUpscaler::SourceImage GSTextureUpscaler::DecodeSource(const std::string& path)
{
	SourceImage src;

	const GSTextureReplacements::ReplacementTextureLoader loader = GSTextureReplacements::GetLoader(path);
	GSTextureReplacements::ReplacementTexture tex;
	if (!loader || !loader(path, &tex, true) || tex.format != GSTexture::Format::Color)
		return src;

	src.width = tex.width;
	src.height = tex.height;
	src.alpha.resize(static_cast<size_t>(tex.width) * tex.height);
	for (u32 y = 0; y < tex.height; y++)
	{
		const u8* row = tex.data.data() + (static_cast<size_t>(y) * tex.pitch);
		for (u32 x = 0; x < tex.width; x++)
			src.alpha[(static_cast<size_t>(y) * tex.width) + x] = row[(x * 4) + 3];
	}
	src.valid = true;
	return src;
}

// Bilinearly resamples a tight single-channel (alpha) image to the given dimensions.
static std::vector<u8> ResampleAlphaBilinear(const std::vector<u8>& src, u32 sw, u32 sh, u32 dw, u32 dh)
{
	std::vector<u8> out(static_cast<size_t>(dw) * dh);
	const float x_ratio = static_cast<float>(sw) / static_cast<float>(dw);
	const float y_ratio = static_cast<float>(sh) / static_cast<float>(dh);
	for (u32 y = 0; y < dh; y++)
	{
		const float sy = std::max(((static_cast<float>(y) + 0.5f) * y_ratio) - 0.5f, 0.0f);
		const u32 sy0 = std::min(static_cast<u32>(sy), sh - 1);
		const u32 sy1 = std::min(sy0 + 1, sh - 1);
		const float fy = sy - static_cast<float>(sy0);
		for (u32 x = 0; x < dw; x++)
		{
			const float sx = std::max(((static_cast<float>(x) + 0.5f) * x_ratio) - 0.5f, 0.0f);
			const u32 sx0 = std::min(static_cast<u32>(sx), sw - 1);
			const u32 sx1 = std::min(sx0 + 1, sw - 1);
			const float fx = sx - static_cast<float>(sx0);
			const float v00 = src[(static_cast<size_t>(sy0) * sw) + sx0];
			const float v10 = src[(static_cast<size_t>(sy0) * sw) + sx1];
			const float v01 = src[(static_cast<size_t>(sy1) * sw) + sx0];
			const float v11 = src[(static_cast<size_t>(sy1) * sw) + sx1];
			const float v = (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy) + (v01 * (1.0f - fx) + v11 * fx) * fy;
			out[(static_cast<size_t>(y) * dw) + x] = static_cast<u8>(v + 0.5f);
		}
	}
	return out;
}

// Loads the model's (opaque) output and restores the original's exact alpha over it (the model
// never sees real PS2 alpha - see SourceImage). The model's color passes through unchanged.
bool GSTextureUpscaler::PostprocessAndSave(const std::string& model_output_path, const SourceImage& source, const std::string& save_path)
{
	const GSTextureReplacements::ReplacementTextureLoader loader = GSTextureReplacements::GetLoader(model_output_path);
	GSTextureReplacements::ReplacementTexture tex;
	if (!loader || !loader(model_output_path, &tex, true) || tex.format != GSTexture::Format::Color)
		return false;

	// The model output was produced from a forced-opaque copy, so restore the original alpha,
	// bilinearly resampled to the model's dimensions.
	const std::vector<u8> orig_alpha = ResampleAlphaBilinear(source.alpha, source.width, source.height, tex.width, tex.height);
	for (u32 y = 0; y < tex.height; y++)
	{
		u8* row = tex.data.data() + (static_cast<size_t>(y) * tex.pitch);
		const u8* alpha_row = orig_alpha.data() + (static_cast<size_t>(y) * tex.width);
		for (u32 x = 0; x < tex.width; x++)
			row[(x * 4) + 3] = alpha_row[x];
	}

	return GSTextureReplacements::SavePNGImage(save_path, tex.width, tex.height, tex.data.data(), tex.pitch);
}

// Prefix for staging directories in the OS temp dir; suffixed with the owning PID (see below).
static constexpr const char* WORK_DIR_PREFIX = "pcsx2-texture-upscale-";

static unsigned int GetCurrentPid()
{
#ifdef _WIN32
	return static_cast<unsigned int>(GetCurrentProcessId());
#else
	return static_cast<unsigned int>(getpid());
#endif
}

// Returns the OS temp directory, or empty if unavailable.
static std::string GetTempBaseDirectory()
{
	const char* tmp = std::getenv("TEMP");
	if (!tmp)
		tmp = std::getenv("TMP");
	if (!tmp)
		tmp = std::getenv("TMPDIR");
#ifndef _WIN32
	if (!tmp)
		tmp = "/tmp";
#endif

	if (tmp && FileSystem::DirectoryExists(tmp))
		return tmp;
	return std::string();
}

// Returns a scratch directory for batch staging, preferring the OS temp dir. Keeping the heavy
// staging churn OUT of the textures tree matters when that tree lives on a cloud-synced folder
// (e.g. OneDrive), where every write triggers sync activity that slows unrelated file
// operations in the same tree - including ones made from the GS thread.
static std::string GetBatchWorkDirectory(const std::string& fallback_dir)
{
	// Namespace the staging directory by PID: RunUpscalerBatch wipes and recreates it at the start
	// of every batch, so a shared name would let two PCSX2 instances upscaling at once delete each
	// other's staging mid-run.
	const std::string subdir(fmt::format("{}{}", WORK_DIR_PREFIX, GetCurrentPid()));

	const std::string tmp = GetTempBaseDirectory();
	if (!tmp.empty())
		return Path::Combine(tmp, subdir);

	return Path::Combine(fallback_dir, "." + subdir);
}

static bool IsProcessAlive(unsigned int pid)
{
#ifdef _WIN32
	const wil::unique_handle handle(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid)));
	if (handle)
		return true;
	// Access denied means the process exists but belongs to another user.
	return (GetLastError() == ERROR_ACCESS_DENIED);
#else
	return (kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM);
#endif
}

// Deletes staging directories left behind by PCSX2 instances that are no longer running - a
// crash or kill orphans them, and (on Windows at least) the OS never purges its temp directory.
// Runs on the worker thread at startup, before any batches.
static void CleanupStaleWorkDirectories()
{
	const std::string tmp = GetTempBaseDirectory();
	if (tmp.empty())
		return;

	FileSystem::FindResultsArray dirs;
	if (!FileSystem::FindFiles(tmp.c_str(), fmt::format("{}*", WORK_DIR_PREFIX).c_str(), FILESYSTEM_FIND_FOLDERS, &dirs))
		return;

	for (const FILESYSTEM_FIND_DATA& fd : dirs)
	{
		// Names look like "pcsx2-texture-upscale-<pid>-stage<N>".
		std::string_view name(Path::GetFileName(fd.FileName));
		name.remove_prefix(std::strlen(WORK_DIR_PREFIX));
		if (const size_t dash = name.find('-'); dash != std::string_view::npos)
			name = name.substr(0, dash);

		const std::optional<unsigned int> pid = StringUtil::FromChars<unsigned int>(name);
		if (!pid.has_value() || pid.value() == GetCurrentPid() || IsProcessAlive(pid.value()))
			continue;

		FileSystem::RecursiveDeleteDirectory(fd.FileName.c_str());
	}
}

// Rename, falling back to copy+delete for cross-volume moves.
static bool MoveFileRobust(const std::string& src, const std::string& dst)
{
	if (FileSystem::RenamePath(src.c_str(), dst.c_str()))
		return true;
	if (FileSystem::CopyFilePath(src.c_str(), dst.c_str(), true))
	{
		FileSystem::DeleteFilePath(src.c_str());
		return true;
	}
	return false;
}

// True if dir contains at least one file. Used between multi-pass invocations to stop the chain
// early once a pass has produced nothing to feed the next one.
static bool DirectoryHasAnyFiles(const std::string& dir)
{
	FileSystem::FindResultsArray files;
	return FileSystem::FindFiles(dir.c_str(), "*", FILESYSTEM_FIND_FILES, &files) && !files.empty();
}

// Deletes every "-stageN" directory for this work_dir, not just the range the current batch is
// about to (re)create. A batch that crashed mid-run with a HIGHER pass count than a later batch
// would otherwise leave its higher-numbered stage dirs orphaned indefinitely: since work_dir is
// derived purely from our own PID, CleanupStaleWorkDirectories() (run once at worker startup)
// intentionally skips directories under our own PID, trusting each batch to manage its own -
// this glob sweep, run before every batch, is what actually keeps that promise regardless of
// which pass count a given batch happens to use.
static void DeleteAllStageDirs(const std::string& work_dir)
{
	const std::string parent(Path::GetDirectory(work_dir));
	const std::string basename(Path::GetFileName(work_dir));
	FileSystem::FindResultsArray dirs;
	if (FileSystem::FindFiles(parent.c_str(), fmt::format("{}-stage*", basename).c_str(), FILESYSTEM_FIND_FOLDERS, &dirs))
	{
		for (const FILESYSTEM_FIND_DATA& fd : dirs)
			FileSystem::RecursiveDeleteDirectory(fd.FileName.c_str());
	}
}

// Runs one upscaler process over a whole directory of textures. The AI model is loaded once
// for the entire batch instead of once per texture, which is dramatically faster.
//
// Multi-pass (job.passes > 1) chains that per-directory invocation: pass N's output directory
// becomes pass N+1's input directory, so the model runs over its own (now much larger) output.
// Each pass gets its OWN directory rather than reusing "-in"/"-out" in place - Upscayl's own
// "Double Upscayl" shipped a bug (upscayl/upscayl#485) where the final large output got
// overwritten by an intermediate pass's file because they shared a path; distinct stageN
// directories make that class of bug structurally impossible here.
u32 GSTextureUpscaler::RunUpscalerBatch(const std::vector<Job>& jobs, std::vector<std::string>* completed)
{
	const std::string work_dir(GetBatchWorkDirectory(std::string(Path::GetDirectory(jobs.front().input))));
	const int passes = std::clamp(jobs.front().passes, 1, MAX_UPSCALER_PASSES);

	std::vector<std::string> stage_dirs;
	stage_dirs.reserve(static_cast<size_t>(passes) + 1);
	for (int i = 0; i <= passes; i++)
		stage_dirs.push_back(fmt::format("{}-stage{}", work_dir, i));

	// Clean up any stale temp dirs from a previous crash/kill - ALL of them, not just the range
	// this batch is about to (re)create - then create fresh ones.
	DeleteAllStageDirs(work_dir);
	for (const std::string& dir : stage_dirs)
	{
		if (!FileSystem::CreateDirectoryPath(dir.c_str(), false))
		{
			Console.Error("Failed to create texture upscale batch directories.");
			for (const std::string& d : stage_dirs)
				FileSystem::RecursiveDeleteDirectory(d.c_str());
			return 0;
		}
	}

	const std::string& in_dir = stage_dirs.front();
	const std::string& out_dir = stage_dirs.back();

	// Stage the pending inputs as opaque copies so the model only sees opaque color data. The dumps
	// dir also contains textures that were already upscaled; staging only what's pending avoids
	// redoing them. staged_opaque[i] records whether the input decoded (opaque copy) or had to be
	// staged raw; the originals are NOT retained here - they're re-decoded per file in the harvest
	// pass below, so a large batch doesn't pin every source image in memory at once.
	std::vector<bool> staged_opaque(jobs.size());
	for (size_t i = 0; i < jobs.size(); i++)
	{
		const std::string staged(Path::Combine(in_dir, Path::GetFileName(jobs[i].input)));
		staged_opaque[i] = StageOpaqueInput(jobs[i].input, staged);
		if (!staged_opaque[i])
		{
			// Couldn't decode it; hand the raw file to the model as a last resort.
			FileSystem::CopyFilePath(jobs[i].input.c_str(), staged.c_str(), true);
		}
	}

	// Run each pass in turn, feeding the previous pass's output directory in as this pass's input.
	// A file that a pass fails to produce simply isn't present in that directory, so it drops out
	// of every subsequent pass and out of the final harvest below - partial failures anywhere in
	// the chain degrade gracefully to "that texture didn't get upscaled this time", same as a
	// single-pass failure always has.
	bool process_ok = true;
	for (int pass = 0; pass < passes; pass++)
	{
		process_ok = RunProcessAndWait(jobs.front().exe_path, BuildUpscalerArgs(jobs.front(), stage_dirs[pass], stage_dirs[pass + 1])) && process_ok;
		if (s_stop.load(std::memory_order_relaxed))
			break;

		// A failed/misconfigured run (e.g. bad executable path) leaves the next stage directory
		// empty - don't spawn further passes over nothing, that's just more wasted processes and
		// duplicate error logs for what's already a failed batch.
		if (pass + 1 < passes && !DirectoryHasAnyFiles(stage_dirs[pass + 1]))
			break;
	}

	// The replacements directory might not exist (it's only auto-created alongside the dumps
	// directory); moving files into a missing directory silently fails.
	FileSystem::EnsureDirectoryExists(std::string(Path::GetDirectory(jobs.front().output)).c_str(), false);

	// Harvest whatever completed, even if a pass failed/was killed partway.
	u32 ok_count = 0;
	for (size_t i = 0; i < jobs.size(); i++)
	{
		const Job& job = jobs[i];
		const std::string filename(Path::GetFileName(job.input));
		const std::string produced(Path::Combine(out_dir, filename));
		if (!FileSystem::FileExists(produced.c_str()))
			continue;

		// Re-decode the original (only if it staged as opaque) to restore its color/alpha.
		SourceImage source;
		if (staged_opaque[i])
			source = DecodeSource(job.input);

		bool ok;
		if (source.valid)
		{
			// Restore the original's colors/alpha into the model's output, then move into place.
			const std::string merged(produced + ".merged.png");
			ok = PostprocessAndSave(produced, source, merged) && MoveFileRobust(merged, job.output);
		}
		else
		{
			// Raw-staged (or no longer decodable): take the model's output as-is - but decode it
			// first as a validity check. A process killed mid-write (e.g. on shutdown) can leave
			// a truncated file, and moving that into place would permanently poison this
			// replacement (it would fail to load every session). The opaque-staged path gets the
			// same check for free via PostprocessAndSave's decode.
			ok = DecodeSource(produced).valid && MoveFileRobust(produced, job.output);
		}

		if (ok)
		{
			DevCon.WriteLnFmt("Upscaled texture '{}'.", Path::GetFileName(job.output));
			completed->push_back(job.output);
			ok_count++;
		}
	}

	if (!process_ok && ok_count != jobs.size())
		Console.WarningFmt("Texture upscaler batch process failed ({}/{} textures completed).", ok_count, jobs.size());

	for (const std::string& dir : stage_dirs)
		FileSystem::RecursiveDeleteDirectory(dir.c_str());
	return ok_count;
}

// Catch-up scan (runs on the worker thread): lists the dumps directory and queues an upscale for
// every dump that doesn't already have a matching replacement, so dumps left un-upscaled by an
// interrupted session get retried. QueueUpscale de-dupes and skips existing outputs.
void GSTextureUpscaler::ScanDirectoryAndEnqueue(const std::string& dumps_dir, const std::string& replacements_dir)
{
	if (!FileSystem::DirectoryExists(dumps_dir.c_str()))
		return;

	// List the replacements directory ONCE and compare in memory. A FileExists() per dump would be
	// hundreds of filesystem round-trips on slow (e.g. cloud-synced) folders. Compare by file
	// title (name minus extension) so a user-provided replacement in another format (e.g. .dds)
	// also suppresses upscaling its dump - otherwise our .png would land next to it and shadow
	// it nondeterministically on the next full replacement-map reload.
	FileSystem::FindResultsArray files;
	std::unordered_set<std::string> existing_replacements;
	if (FileSystem::FindFiles(replacements_dir.c_str(), "*", FILESYSTEM_FIND_FILES, &files))
	{
		existing_replacements.reserve(files.size());
		for (FILESYSTEM_FIND_DATA& fd : files)
			existing_replacements.emplace(Path::GetFileTitle(fd.FileName));
	}

	FileSystem::FindFiles(dumps_dir.c_str(), "*.png", FILESYSTEM_FIND_FILES, &files);
	for (const FILESYSTEM_FIND_DATA& fd : files)
	{
		// Bail promptly if the feature was turned off (or we're shutting down) mid-scan.
		if (!IsEnabled())
			break;

		const std::string_view filename(Path::GetFileName(fd.FileName));
		if (existing_replacements.find(std::string(Path::GetFileTitle(filename))) == existing_replacements.end())
			QueueUpscale(fd.FileName, Path::Combine(replacements_dir, filename), true);
	}
}
