#include <bits/stdc++.h>
#define db double
#define ll long long
#define vi vector<int>
#define vii vector<vi >
#define vd vector<db>
#define vdd vector<vd >
#define pii pair<int, int>
#define pdd pair<db, db>
#define vpd vector<pdd >
#define vipd vector<vpd >
#define vp vector<pii >
#define vip vector<vp >
#define mkp make_pair
#define pb push_back
using namespace std;
const int INF = 0x3f3f3f3f;
const int T = 60000; // Number of Total training Data
const int L = 4; // Number of Layers (contains Input layer and Output layer)
const int IN = 784; // Number of Nodes in Layer 1 (Input Layer)
const int OUT = 10; // Number of Nodes in Layer L-1 (Output Layer)
const int N[L] = {IN, 16, 16, OUT}; // Number of Nodes in each Layer
//vd N(L); 
db image[T][IN]; // Image Data
int ans[T]; // Label of Image Data (Answer)
const int GROUP = 100; // Learning Group (Upgrade the network by GROUP numbers of Learning Data)
const int TOT = 100000; // Number of ANN, 1 7s
const int THR = 8; // Number of Threads
struct mat{ // Matrix Data Struct
	int n, m; // Size of Matrix : n * m
	vdd M;
	mat() {}
	mat(int n, int m, int num = 0) : n(n), m(m) { M = vdd(n, vd(m, num)); }
	mat operator * (const mat &y) const & { // multiply of Matrix
		assert(m == y.n);
		mat z(n, y.m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < y.m; j++)
				for (int k = 0; k < m; k++)
					z.M[i][j] += M[i][k] * y.M[k][j];
		return z;
	}
	mat operator + (const mat &y) const & { // addition of Matrix
		assert(n == y.n && m == y.m);
		mat z(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				z.M[i][j] = M[i][j] + y.M[i][j];
		return z;
	}
	mat operator * (const double &y) const & { // multiply Matrix and Const
		mat z(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				z.M[i][j] = M[i][j] * y;
		return z;
	}
	mat dot(mat &x, mat &y) { // dot multiplay of Matrix
		assert(x.n == y.n && x.m == y.m);
		int n = x.n, m = x.m;
		mat z(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				z.M[i][j] = x.M[i][j] * y.M[i][j];
		return z;
	}
	mat operator ~ () const & { // transpose the Matrix
		mat z(m, n);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				z.M[i][j] = M[j][i];
		return z;
	}
	void print() { // print the Matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				printf("%.2lf ", M[i][j]);
			}
			putchar('\n');
		}
		putchar('\n');
	}
}MAT;
struct layer { // Layer of the Network
	mat a, w, b, z;
	int id;
	layer() {}
	layer(int id) : id(id) {
		a = mat(N[id], 1);
		if (id) {
			w = mat(N[id], N[id-1]);
			b = mat(N[id], 1);
		}
	}
}baseNet[L]; // basic network
db getrand() { return 1.0 * rand() / RAND_MAX; }
void init() { // initialize Training Data
	freopen("train.in", "r", stdin);
	for (int i = 0; i < T; i++) {
		for (int j = 0; j < IN; j++) {
			scanf("%lf", &image[i][j]);
			image[i][j] /= 255;
		}
	}
	fclose(stdin);
	freopen("train.out", "r", stdin);
	for (int i = 0; i < T; i++) scanf("%d", &ans[i]);
	fclose(stdin);
	printf("Reading complete!\n");
	// image Input TEST
	//for (int i = 0; i < 784; i++) {
	//	printf("%d ", (int)(image[0][i] * 255));
	//	if ((i+1) % 28 == 0) {
	//		putchar('\n');
	//	}
	//}
	freopen("diary.out", "w", stdout);
	fclose(stdout);
}
void Save(int num) { // Print Learning Result
	string s = string("Result") + to_string(num) + string(".out");
	freopen(s.c_str(), "w", stdout);
	printf("%d\n", L);
	for (int i = 0; i < L; i++) printf("%d ", N[i]);
	putchar('\n');
	for (int l = 1; l < L; l++) {
		baseNet[l].w.print();
		baseNet[l].b.print();
	}
	fclose(stdout);
}
// Back Propagation (Learning)
mat f(mat &x) { // activate function (sigmoid)
	mat z(x.n, x.m);
	for (int i = 0; i < x.n; i++) {
		for (int j = 0; j < x.m; j++) {
			db t = x.M[i][j];
			z.M[i][j] = 1.0 / (1 + exp(-t));
		}
	}
	return z;
}
mat _f(mat &x) { // Derivative of activate function 
	mat z(x.n, x.m);
	for (int i = 0; i < x.n; i++) {
		for (int j = 0; j < x.m; j++) {
			db t = x.M[i][j];
			z.M[i][j] = 1.0 / (exp(t) + exp(-t) + 2);
		}
	}
	return z;
}
// Id of Learning Data and Total gradient and cost
db BP(int id, layer grad[]) { // return Cost
	layer net[L];
	mat y(OUT, 1); // Desired result (Answer)
	for (int i = 0; i < L; i++) net[i] = baseNet[i];
	// initialize Input & Desired Data
	for (int i = 0; i < IN; i++) net[0].a.M[i][0] = image[id][i];
	for (int i = 0; i < OUT; i++)
		if (i == ans[id]) y.M[i][0] = 1;
	// Forward
	for (int l = 1; l < L; l++) {
		net[l].z = net[l].w * net[l-1].a + net[l].b;
		net[l].a = f(net[l].z);
	}
	// Backward
	mat dc_da = (net[L-1].a + (y * (-1))) * 2;
	for (int l = L-1; l >= 1; l--) {
		mat _fz = _f(net[l].z);
		mat dc_db = MAT.dot(dc_da, _fz);
		grad[l].b = grad[l].b + dc_db;
		grad[l].w = grad[l].w + (dc_db * (~net[l-1].a));
		dc_da = (~net[l].w) * dc_db;
	}
	// Cost
	db cost = 0;
	for (int i = 0; i < OUT; i++) cost += pow(net[L-1].a.M[i][0] - y.M[i][0], 2);
	return cost;
}
void GroupLearn(int st, vi *perm, layer grad[], db *cost) { // Thread of Group Learning with start id
	for (int j = st; j < st + GROUP; j++) { // assign Learning tasks
		*cost += BP((*perm)[j], grad);
	}
}
void ANN() { // Artificial Neural Network
	// initialize the struct of Network
	for (int i = 0; i < L; i++) baseNet[i] = layer(i);
	if (freopen("Result.in", "r", stdin) == NULL) { // initialize w and b randomly
		freopen("/dev/tty", "w", stdout);
		printf("Randomly initialization\n");
		for (int l = 1; l < L; l++) {
			for (int i = 0; i < N[l]; i++) {
				for (int j = 0; j < N[l-1]; j++)
					baseNet[l].w.M[i][j] = getrand() * 10 - 5;
				baseNet[l].b.M[i][0] = getrand() * 40 - 20;
			}
		}
	} else { // Using last Learning Data
		freopen("/dev/tty", "w", stdout);
		printf("Get Result.in\n");
		int rL;
		scanf("%d", &rL);
		assert(L == rL);
		vi rN(L);
		for (int i = 0; i < L; i++) {
			scanf("%d", &rN[i]);
			assert(rN[i] == N[i]);
		}
		for (int l = 1; l < L; l++) {
			for (int i = 0; i < N[l]; i++)
				for (int j = 0; j < N[l-1]; j++)
					scanf("%lf", &baseNet[l].w.M[i][j]);
			for (int i = 0; i < N[l]; i++)
				scanf("%lf", &baseNet[l].b.M[i][0]);
		}
		fclose(stdin);
	}
	vi perm(T);
	for (int i = 0; i < T; i++) perm[i] = i;
	int fg = 0; // id of Save data
	for (int _i = 0; _i < TOT; _i++) {
		random_shuffle(perm.begin(), perm.end());
		db Cost = 0;
		for (int i = 0; i < T; i += GROUP * THR) {
			layer grad[THR][L]; // average gradient of a group for each thread
			db cost[THR] = {0}; // average cost of a group
			thread th[THR];
			for (int t = 0; t < THR; t++)
				for (int l = 0; l < L; l++)
					grad[t][l] = layer(l);
			for (int t = 0; t < THR; t++) {
				th[t] = thread(GroupLearn, i + t * GROUP, &perm, grad[t], &cost[t]);
			}
			for (int t = 0; t < THR; t++) {
				th[t].join();
			}
			for (int i = 1; i < L; i++) { // Upgrade Network
				for (int t = 1; t < THR; t++) {
					grad[0][i].w = grad[0][i].w + grad[t][i].w;
					grad[0][i].b = grad[0][i].b + grad[t][i].b;
				}
			}
			for (int i = 1; i < L; i++) {
				baseNet[i].w = baseNet[i].w + grad[0][i].w * (-1.0 / (GROUP * THR));
				baseNet[i].b = baseNet[i].b + grad[0][i].b * (-1.0 / (GROUP * THR));
			}
			for (int t = 0; t < THR; t++) {
				Cost += cost[t] / GROUP;
			}
		}
		Save(fg);
		fg ^= 1;
		freopen("diary.out", "a", stdout);
		printf("%lf\n", Cost / (T / GROUP));
		fclose(stdout);
		freopen("/dev/tty", "w", stdout);
		printf("complete turn: %d\n", _i+1);
	}
	// TEST
	//for (int i = 0; i < 10; i++) {
	//	layer grad[L]; // average gradient of the group
	//	for (int l = 0; l < L; l++) grad[l] = layer(l);
	//	for (int j = i; j < i + 1; j++) { // assign Learning tasks
	//		db cost = BP(perm[0], grad);
	//		printf("%lf\n", cost);
	//	}
	//	for (int i = 1; i < L; i++) { // Upgrade Network
	//		baseNet[i].w = baseNet[i].w + grad[i].w * (-1.0 / GROUP);
	//		baseNet[i].b = baseNet[i].b + grad[i].b * (-1.0 / GROUP);
	//	}
	//}
}
signed main() {
	srand(time(NULL));
	init();
	clock_t st = clock(), en;
	ANN();
	en = clock();
	freopen("diary.out", "a", stdout);
	printf("Learning time: %lf s\n", 1.0 * (en - st) / CLOCKS_PER_SEC);
	fclose(stdout);
	return 0;
}
