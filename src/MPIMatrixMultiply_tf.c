// matrix multiplication using Task Farming strategy
// master distribute tasks to workers dynamically and collect results
// A x B = result. A(N * K), B(K * M), C(N * M); Number of process = P
// communication costs ~ 2 * O(N*M) i.e. the number of tasks
// One time const of a massage: t_msg = L + M / B, where L is the latency, M is the message size, B is the bandwidth
// T_tot = O(N*M) * (L + M / B) + O(N*K*M / P) , RHS second item for computation
// Main disadvantage: Communication overhead, master can become a bottleneck, frequently pack msg and send through network
// Main advantage: Dynamic load balancing. 
// Suitable when:
// computation time of a processor is unpredictable, 
// computation time is far more than communication time, 
// or the memory is limited.