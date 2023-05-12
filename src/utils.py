import time

def execution_time(_function):
	start_time = time.time()

	response = _function()

	finish_time = time.time()

	total_time = finish_time - start_time
	return response, total_time