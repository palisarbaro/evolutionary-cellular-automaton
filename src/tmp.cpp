
/*using namespace sycl;
static auto exception_handler = [](sycl::exception_list e_list) {
	for (std::exception_ptr const &e : e_list) {
		try {
			std::cout << "EEERRRR" << std::endl;
			//std::rethrow_exception(e);
		}
		catch (std::exception const &e) {
			std::cout << "Failure" << std::endl;
			//std::terminate();
		}
	}
};
float getNorm(int N, float* x) {
	float res = -1100000000.;
	for (int i = 0; i < N; i++) {
		if (res < fabs(x[i])) {
			res = fabs(x[i]);
		}
	}
	
	return res;
}
float check(std::vector<float> A, std::vector<float> b, std::vector<float> x) {
	std::vector<float>diff;
	for (int i = 0; i < b.size(); i++) {
		float sum = 0;
		for (int j = 0; j < b.size(); j++) {
			sum += A[j*b.size() + i]*x[j];
		}
		diff.push_back(sum - b[i]);
	}
	return getNorm(b.size(),&diff[0]);
}

void jacobiBuffer(sycl::device device, int N, std::vector<float> A, std::vector<float> b, float accuracy, int maxIterations) {
	std::cout << "jacobi(buffers) on device " << device.get_info<info::device::name>() << std::endl;
	property_list props{ property::queue::enable_profiling() };
	queue gpu_queue(device, exception_handler, props);
	uint64_t time=0;
	std::vector<float> x1(N,0.0);
	std::vector<float> x2(N,0.0);
	std::vector<float> diff(N,10000.0);
	{
		
		buffer<float> buf_A(A.data(), A.size());
		buffer<float> buf_b(b.data(), b.size());
		buffer<float> buf_x1(x1.data(), x1.size());
		buffer<float> buf_x2(x2.data(), x2.size());
		auto buff_xk = &buf_x1;
		auto buff_xkp1 =&buf_x2;
		buffer<float> buf_diff(diff.data(), diff.size());
		for (int iterations = 0; iterations < maxIterations; iterations++) {
			event e = gpu_queue.submit([&](handler& cgh) {
				auto A = buf_A.get_access<access::mode::read>(cgh);
				auto b = buf_b.get_access<access::mode::read>(cgh);
				auto x1 = buff_xk->get_access<access::mode::read>(cgh);
				auto x2 = buff_xkp1->get_access<access::mode::write>(cgh);
				auto diff = buf_diff.get_access<access::mode::write>(cgh);

				cgh.parallel_for(range<1>(N), [=](id<1> item) {

					x2[item] = b[item];
					for (int j = 0; j < N; j++) {
						x2[item] -= A[item + j * N] * x1[j];
					}
					x2[item] += A[item + item * N] * x1[item];
					x2[item] /= A[item + item * N];
					diff[item] = x1[item] - x2[item];

				});

			});

			e.wait_and_throw();

			auto start = e.get_profiling_info<info::event_profiling::command_start>();
			auto end = e.get_profiling_info<info::event_profiling::command_end>();
			time += end - start;
			auto tmp = buff_xk;
			buff_xk = buff_xkp1;
			buff_xkp1 = tmp;
			auto host_diff = buf_diff.get_host_access();
			auto host_x1 = buff_xk->get_host_access();

			if (getNorm(N, host_diff.get_pointer()) / getNorm(N, host_x1.get_pointer()) < accuracy) {
				std::cout << "iterations: " << iterations << std::endl;
				break;
			}



		}
	}

	std::cout << "acc: "<< check(A, b, x1) << std::endl;
	std::cout << time / 1000000. << " ms" << std::endl;

}
void jacobiShared(sycl::device device, int N, std::vector<float> A, std::vector<float> b, float accuracy, int maxIterations) {
	std::cout << "jacobi(shared) on device " << device.get_info<info::device::name>() << std::endl;

	property_list props{ property::queue::enable_profiling() };
	queue gpu_queue(device, exception_handler, props);
	float* xk = sycl::malloc_shared<float>(N, gpu_queue);
	float* xkp1 = sycl::malloc_shared<float>(N, gpu_queue);
	float* diff = sycl::malloc_shared<float>(N, gpu_queue);
	float* A_shared = sycl::malloc_shared<float>(N*N, gpu_queue);
	float* b_shared = sycl::malloc_shared<float>(N, gpu_queue);

	uint64_t time = 0;


	for (int i = 0; i < N; i++) {
		b_shared[i] = b[i];
		xk[i] = 0;
		for (int j = 0; j < N; j++) {
			A_shared[i + j * N] = A[i + j * N];
		}
	}
	for (int iteration = 0; iteration < maxIterations; iteration++) {
		sycl::event e = gpu_queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
			xkp1[i] = b_shared[i];
			for (int j = 0; j < N; j++) {
				xkp1[i] -= A_shared[i + j * N] * xk[j];
			}
			xkp1[i] += A_shared[i + i * N] * xk[i];
			xkp1[i] /= A_shared[i + i * N];
			diff[i] = xk[i] - xkp1[i];
		});
		e.wait_and_throw();
		auto start = e.get_profiling_info<info::event_profiling::command_start>();
		auto end = e.get_profiling_info<info::event_profiling::command_end>();
		time += end - start;

		float* tmp = xk;
		xk = xkp1;
		xkp1 = tmp;
		if (getNorm(N, diff) / getNorm(N, xkp1) < accuracy) {
			std::cout << "iterations: " << iteration << std::endl;
			break;
		}
	}
	std::vector<float> x(N, 0.0);
	for (int i = 0; i < N; i++) {
		x[i] = xk[i];
	}
	std::cout << "acc: " << check(A, b, x) << std::endl;
	std::cout << "time: " << time / 1000000. << " ms" << std::endl;
	sycl::free(xk,gpu_queue);
	sycl::free(xkp1,gpu_queue);
	sycl::free(diff,gpu_queue);
	sycl::free(A_shared,gpu_queue);
	sycl::free(b_shared,gpu_queue);

}
void jacobiDevice(sycl::device device, int N, std::vector<float> A, std::vector<float> b, float accuracy, int maxIterations) {
	std::cout << "jacobi(device) on device " << device.get_info<info::device::name>() << std::endl;
	std::vector<float> diff(N,100000);
	std::vector<float> x(N,100000);
	property_list props{ property::queue::enable_profiling() };
	queue gpu_queue(device, exception_handler, props);
	float* xk = sycl::malloc_device<float>(N, gpu_queue);
	float* xkp1 = sycl::malloc_device<float>(N, gpu_queue);
	float* diff_shared = sycl::malloc_device<float>(N, gpu_queue);
	float* A_shared = sycl::malloc_device<float>(N*N, gpu_queue);
	float* b_shared = sycl::malloc_device<float>(N, gpu_queue);

	uint64_t time = 0;
	gpu_queue.memcpy(A_shared, A.data(), N*N*sizeof(float)).wait();
	gpu_queue.memcpy(b_shared, b.data(), N*sizeof(float)).wait();

	for (int iteration = 0; iteration < maxIterations; iteration++) {
		sycl::event e = gpu_queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
			xkp1[i] = b_shared[i];
			for (int j = 0; j < N; j++) {
				xkp1[i] -= A_shared[i + j * N] * xk[j];
			}
			xkp1[i] += A_shared[i + i * N] * xk[i];
			xkp1[i] /= A_shared[i + i * N];
			diff_shared[i] = xk[i] - xkp1[i];
		});
		e.wait_and_throw();
		auto start = e.get_profiling_info<info::event_profiling::command_start>();
		auto end = e.get_profiling_info<info::event_profiling::command_end>();
		time += end - start;

		gpu_queue.memcpy(diff.data(), diff_shared, N * sizeof(float)).wait();
		gpu_queue.memcpy(x.data(), xk, N * sizeof(float)).wait();

		float* tmp = xk;
		xk = xkp1;
		xkp1 = tmp;
		if (getNorm(N, &diff[0]) / getNorm(N, &x[0]) < accuracy) {
			std::cout << "iterations: " << iteration << std::endl;
			break;
		}
	}

	std::cout << "acc: " << check(A, b, x) << std::endl;
	std::cout << "time: " << time / 1000000. << " ms" << std::endl;
	sycl::free(xk, gpu_queue);
	sycl::free(xkp1, gpu_queue);
	sycl::free(diff_shared, gpu_queue);
	sycl::free(A_shared, gpu_queue);
	sycl::free(b_shared, gpu_queue);
	

}
float getRandFloat() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

std::vector<float> genMatrix(int N) {
	std::vector<float> res;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res.push_back(getRandFloat()*2-1);
			if (i == j) {
				res[res.size() - 1] = N*2;
			}
			//std::cout << res[res.size() - 1] << ", ";
		}
	}
	//std::cout << std::endl;
	return res;
}
std::vector<float> genVector(int N) {
	std::vector<float> res;
	for (int i = 0; i < N; i++) {
		res.push_back(getRandFloat() * 2 - 1);
		//std::cout << res[res.size() - 1] << ", ";

	}
	//std::cout << std::endl;

	return res;
}


	std::vector<sycl::device> devices;
	std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
	int i = 0, j = 0;

	for (auto platform : platforms) {
		std::cout << "Platform#" << i << ": " << platform.get_info<sycl::info::platform::name>() << std::endl;
		std::vector<sycl::device> _devices = platform.get_devices();
		j = 0;
		for (auto device : _devices) {
			std::cout << "   Device#" << j << ": " << device.get_info<sycl::info::device::name>() << std::endl;
			devices.push_back(device);
			j++;
		}
		i++;
	}
	int N = 2000;
	std::vector<float> A = genMatrix(N);
	std::vector<float> b = genVector(N);
	
	auto device = devices[dev=="cpu"?1:0];
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	jacobiBuffer(device, N, A, b, acc, maxIter);
	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	jacobiShared(device, N, A, b, acc, maxIter);
	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
	jacobiDevice(device, N, A, b, acc, maxIter);
	std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

	std::cout << "jacobiBuffer: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	std::cout << "jacobiShared: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;
	std::cout << "jacobiDevice: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;

	*/