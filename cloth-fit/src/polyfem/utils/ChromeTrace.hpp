#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace polyfem::utils
{
	/// Minimal Chrome Trace (catapult) JSON writer.
	/// View output in `chrome://tracing` or Perfetto UI.
	class ChromeTrace
	{
	public:
		static ChromeTrace &instance();

		/// Enable tracing. If already enabled, does nothing.
		bool init(const std::string &path);

		/// Flush and close the file.
		void shutdown();

		bool enabled() const { return enabled_; }

		/// Write a complete event ("X" phase).
		void write_complete_event(
			const std::string &name,
			int64_t ts_us,
			int64_t dur_us,
			uint32_t pid,
			uint64_t tid,
			const std::string &cat = "polyfem");

		/// Timestamp in microseconds since init.
		int64_t now_us() const;

	private:
		ChromeTrace() = default;
		~ChromeTrace();

		void write_event_json_locked(const std::string &json);

		bool enabled_ = false;
		std::ofstream out_;
		mutable std::mutex mtx_;
		bool first_event_ = true;
		std::optional<std::chrono::steady_clock::time_point> t0_;
	};

	/// RAII scope event recorded as a complete ("X") Chrome trace event.
	class ChromeTraceScope
	{
	public:
		explicit ChromeTraceScope(const std::string &name, const std::string &cat = "polyfem");
		~ChromeTraceScope();

	private:
		std::string name_;
		std::string cat_;
		int64_t start_us_ = 0;
		bool active_ = false;
	};
} // namespace polyfem::utils

