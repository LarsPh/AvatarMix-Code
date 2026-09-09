#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace polyfem::utils
{
	struct TimingEntry
	{
		std::string name;
		double total_time_s = 0.0;
		size_t count = 0;
	};

	class TimingRegistry
	{
	public:
		static TimingRegistry &instance();

		void set_enabled(bool enabled);
		bool enabled() const;

		void add(const std::string &name, double dt_s);

		/// Returns entries sorted by decreasing total time.
		std::vector<TimingEntry> snapshot_sorted() const;

		void clear();

	private:
		TimingRegistry() = default;
		~TimingRegistry() = default;
	};
} // namespace polyfem::utils

