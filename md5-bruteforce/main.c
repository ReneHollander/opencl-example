#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else

#include <CL/cl.h>

#endif

#include <math.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)
#define GLOBAL_ITEM_SIZE 8192
#define LOCAL_ITEM_SIZE 64
#define BILLION 1E9

int main(void) {

    int maxlen = 7;

    long permutations = 0;
    for (long i = 1; i <= maxlen; i++) {
        permutations += pow(26, i);
    }
    long permutationsPerThread = permutations / GLOBAL_ITEM_SIZE;
    long missingPermutations = permutations - permutationsPerThread * GLOBAL_ITEM_SIZE;

    printf("Permutations globally: %lu!\n", permutations);
    printf("Permutations per thread: %lu!\n", permutationsPerThread);
    printf("Permutations missing: %lu!\n", missingPermutations);
    printf("\n");

    int *starts = (int *) malloc(sizeof(int) * GLOBAL_ITEM_SIZE);
    int *stops = (int *) malloc(sizeof(int) * GLOBAL_ITEM_SIZE);
    uint *pw_hash = (uint *) malloc(sizeof(uint) * 4);

    sscanf("c75e86c6362f42a5b07cfe0f66d3d10a", "%08x%08x%08x%08x", &pw_hash[0], &pw_hash[1], &pw_hash[2], &pw_hash[3]);
    printf("Input Hash: %x%x%x%x\n", pw_hash[0], pw_hash[1], pw_hash[2], pw_hash[3]);

    int count = 0;
    for (int i = 0; i < GLOBAL_ITEM_SIZE; i++) {
        *(starts + i) = count;
        count += permutationsPerThread;
        *(stops + i) = count;
        count++;
    }
    *(stops + GLOBAL_ITEM_SIZE - 1) += missingPermutations;

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = 0;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    char str[128];
    size_t strSize = (sizeof(char) * 128);
    size_t retSize;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, strSize, (void *) str, &retSize);
    printf("Name: %s\n", str);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem starts_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, GLOBAL_ITEM_SIZE * sizeof(int), NULL, &ret);
    cl_mem stops_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, GLOBAL_ITEM_SIZE * sizeof(int), NULL, &ret);
    cl_mem maxlen_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem pw_hash_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint) * 4, NULL, &ret);
    cl_mem cracked_pw_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * (maxlen + 1), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, starts_mem_obj, CL_TRUE, 0, GLOBAL_ITEM_SIZE * sizeof(int), starts, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, stops_mem_obj, CL_TRUE, 0, GLOBAL_ITEM_SIZE * sizeof(int), stops, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, maxlen_mem_obj, CL_TRUE, 0, sizeof(int), &maxlen, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, pw_hash_mem_obj, CL_TRUE, 0, sizeof(uint) * 4, pw_hash, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &starts_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &stops_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &maxlen_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &pw_hash_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &cracked_pw_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = GLOBAL_ITEM_SIZE; // Process the entire lists
    size_t local_item_size = LOCAL_ITEM_SIZE; // Divide work items into groups of 64
    cl_event event;

    struct timespec requestStart, requestEnd;
    clock_gettime(CLOCK_REALTIME, &requestStart);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clock_gettime(CLOCK_REALTIME, &requestEnd);

    char *cracked_pw = (char *) malloc(sizeof(char) * (maxlen + 1));
    ret = clEnqueueReadBuffer(command_queue, cracked_pw_mem_obj, CL_TRUE, 0, sizeof(char) * (maxlen + 1), cracked_pw, 0, NULL, NULL);
    printf("Cracked Password: %s\n", cracked_pw);
    double accum = (requestEnd.tv_sec - requestStart.tv_sec) + (requestEnd.tv_nsec - requestStart.tv_nsec) / BILLION;
    printf("Time to crack Password: %lfs\n", accum);


    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(starts_mem_obj);
    ret = clReleaseMemObject(stops_mem_obj);
    ret = clReleaseMemObject(maxlen_mem_obj);
    ret = clReleaseMemObject(pw_hash_mem_obj);
    ret = clReleaseMemObject(cracked_pw_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(starts);
    free(stops);
    free(cracked_pw);
    return 0;
}