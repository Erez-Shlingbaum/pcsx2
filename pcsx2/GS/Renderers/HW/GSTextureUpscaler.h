// SPDX-FileCopyrightText: 2002-2025 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include <string>
#include <vector>

// Optional "just-in-time" AI upscaling of dumped textures.
//
// When enabled, dumped textures are handed to an external, user-provided upscaler executable
// (Real-ESRGAN / Upscayl "upscayl-bin" CLI compatible) which is run as a SEPARATE PROCESS.
// The result is written into the game's texture replacement directory, where the existing
// replacement system picks it up. Nothing is linked or bundled - the user supplies the
// executable and models - so this imposes no additional licensing constraints on PCSX2.
namespace GSTextureUpscaler
{
	/// Refreshes cached configuration from EmuConfig. Call whenever GS options change.
	void UpdateConfig();

	/// Returns true if upscaling is enabled and a usable executable path is configured.
	bool IsEnabled();

	/// Queues an asynchronous upscale of dump_path, writing the output to replacement_path.
	/// No-op when disabled, when replacement_path already exists, or when an identical job is
	/// already queued/running. Pass skip_exists_check when the caller has already verified the
	/// output is missing (the check is a filesystem round-trip, which matters on slow folders).
	///
	/// Does synchronous filesystem work (reads dump_path's PNG header for the min-size gate, and -
	/// unless skip_exists_check - stats replacement_path). Only call from the upscaler's own worker
	/// thread or another non-latency-sensitive context, never from the GS thread; use
	/// QueueUpscaleFromDump() from the texture-dump path instead.
	void QueueUpscale(std::string dump_path, std::string replacement_path, bool skip_exists_check = false);

	/// Enqueues dump_path for upscaling with NO synchronous filesystem work: the min-size gate and
	/// the replacement-exists check are deferred to the upscaler's own worker thread. Intended for
	/// the texture-dump work item, which runs on the GSTextureReplacements worker thread that the
	/// GS thread synchronously waits on (SyncWorkerThread) - doing the dump PNG re-read and the
	/// existence stat there (both potentially slow on a cloud-synced folder) would stall the GS
	/// thread and freeze the game. No-op when disabled.
	void QueueUpscaleFromDump(std::string dump_path, std::string replacement_path);

	/// Queues a catch-up scan of a dumps directory: every dump lacking a matching replacement is
	/// queued for upscaling. The slow directory enumeration and per-file header reads run on the
	/// worker thread, not the caller's, so this is cheap to call from the GS thread. No-op when
	/// disabled.
	void QueueDirectoryScan(std::string dumps_dir, std::string replacement_dir);

	/// Returns a limited number of the replacement files completed since the last call. The
	/// caller should register them via GSTextureReplacements::AddReplacementFiles(), which swaps
	/// in-use textures live. Returns false when nothing is ready. Call on the GS thread.
	bool TakeCompletedFiles(std::vector<std::string>* files);

	/// Clears queued-but-not-yet-started jobs, pending scans, and progress/completed-file state.
	/// Does not touch an already in-flight process (it's left to finish on its own) or the worker
	/// thread itself. Call when the game changes, so a new game doesn't inherit stale queued work
	/// - and OSD progress counts - left over from the previous one.
	void CancelPendingJobs();

	/// Cancels queued jobs and terminates/awaits the in-flight process. Call on shutdown.
	void Shutdown();
} // namespace GSTextureUpscaler
