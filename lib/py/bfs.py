
class Finder(object):
	#globle variables
	home = {
		'x', 0,
		'y', 0,
	}
	best_path = None
	max_reward = 0

	@classmethod
	def best_route(cls, current_location, jobs, capacity):
		#initialize
		# dummy_job = {
		# 	'x': current_location['x'],
		# 	'y': current_location['y'],
		# 	'v': 0,
		# }
		tasks = []
		cls.best_path = None
		cls.max_reward = 0
		for job in jobs:
			tasks.append(job)
			cls.dfs(jobs, 0, capacity, tasks, 0)

	@classmethod
	def dfs(cls, jobs, index, capacity, tasks, value):
		'''traverse the tree and update globle result'''
		#current
		current_job = tasks[-1]
		value += current_job['v']
		cls.evaluate(current_job, tasks, value)
		#terminate
		if len(tasks) > capacity or index >= len(jobs):
			return
		#recursive rules
		for i in range(index+1, len(jobs)):
			cls.swap(jobs, index, i)
			tasks.append(jobs[index])
			cls.dfs(jobs, index+1, capacity, tasks, value)
			tasks.pop()
			cls.swap(jobs, index, i)

	@classmethod
	def evaluate(cls, job, tasks, value):
		#return home now
		return_path = cls.shortest_path(1,1)
		current_path = cls.get_action_path(tasks)
		final_path = current_path + return_path
		#compute reward
		reward = 0
		if len(final_path) > 0:
			reward = value / len(final_path)
		#update 
		if reward > cls.max_reward:
			cls.max_reward = reward
			final_path = current_path + return_path
			cls.best_path = final_path

	@classmethod
	def swap(cls, array, left, right):
		tmp = array[left]
		array[left] = array[right]
		array[right] = tmp

	@classmethod
	def get_action_path(cls, tasks):
		for task in tasks:
			pass
		return [1]

	@classmethod
	def shortest_path(cls, loc_from, loc_to):
		return [1]

if __name__ == '__main__':
	#-------------------test
	jobs = []
	for i in range(10):
		jobs.append({
				'x': 0,
				'y': 0,
				'v': 1,
			})
	current_location = {
		'x': 0,
		'y': 0,
	}
	Finder.best_route(current_location, jobs, 6)
	print(Finder.max_reward)
	print(Finder.best_path)
