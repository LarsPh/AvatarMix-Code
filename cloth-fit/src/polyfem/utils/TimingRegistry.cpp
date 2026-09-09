#include "TimingRegistry.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace polyfem::utils
{
	namespace
	{
		struct State
		{
			bool enabled = false;
			std::unordered_map<std::string, TimingEntry> map;
			mutable std::mutex mtx;
		};

		State &state()
		{
			static State s;
			return s;
		}
	} // namespace

	TimingRegistry &TimingRegistry::instance()
	{
		static TimingRegistry inst;
		return inst;
	}

	void TimingRegistry::set_enabled(bool enabled)
	{
		std::lock_guard<std::mutex> lock(state().mtx);
		state().enabled = enabled;
	}

	bool TimingRegistry::enabled() const
	{
		std::lock_guard<std::mutex> lock(state().mtx);
		return state().enabled;
	}

	void TimingRegistry::add(const std::string &name, double dt_s)
	{
		if (name.empty())
			return;
		std::lock_guard<std::mutex> lock(state().mtx);
		if (!state().enabled)
			return;
		auto &e = state().map[name];
		e.name = name;
		e.total_time_s += dt_s;
		e.count += 1;
	}

	std::vector<TimingEntry> TimingRegistry::snapshot_sorted() const
	{
		std::lock_guard<std::mutex> lock(state().mtx);
		std::vector<TimingEntry> entries;
		entries.reserve(state().map.size());
		for (const auto &kv : state().map)
			entries.push_back(kv.second);

		std::sort(entries.begin(), entries.end(), [](const TimingEntry &a, const TimingEntry &b) {
			return a.total_time_s > b.total_time_s;
		});
		return entries;
	}

	void TimingRegistry::clear()
	{
		std::lock_guard<std::mutex> lock(state().mtx);
		state().map.clear();
	}
} // namespace polyfem::utils

