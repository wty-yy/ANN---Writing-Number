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
int T; // Number of Total training Data
const int L = 4; // Number of Layers (contains Input layer and Output layer)
const int IN = 784; // Number of Nodes in Layer 1 (Input Layer)
const int OUT = 10; // Number of Nodes in Layer L-1 (Output Layer)
const int N[L] = {IN, 16, 16, OUT}; // Number of Nodes in each Layer
//vd N(L); 
vdd image; // Image Data
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
	freopen("mytest.in", "r", stdin);
	scanf("%d", &T);
	//T = 10;
	image = vdd(T, vd(IN));
	for (int i = 0; i < T; i++) {
		for (int j = 0; j < IN; j++) {
			scanf("%lf", &image[i][j]);
			image[i][j] /= 255;
		}
	}
	fclose(stdin);
	printf("Reading complete!\n");
	// image Input TEST
	//for (int i = 0; i < 784; i++) {
	//	printf("%d ", (int)(image[0][i] * 255));
	//	if ((i+1) % 28 == 0) {
	//		putchar('\n');
	//	}
	//}
}
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
// Id of Checking Data
int CK(int id) { // return Output
	layer net[L];
	for (int i = 0; i < L; i++) net[i] = baseNet[i];
	// initialize Input & Desired Data
	for (int i = 0; i < IN; i++) net[0].a.M[i][0] = image[id][i];
	// Forward
	for (int l = 1; l < L; l++) {
		net[l].z = net[l].w * net[l-1].a + net[l].b;
		net[l].a = f(net[l].z);
	}
	double mx = 0;
	int out;
	for (int i = 0; i < OUT; i++) {
		if (net[L-1].a.M[i][0] > mx) {
			mx = net[L-1].a.M[i][0];
			out = i;
		}
	}
	return out;
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
	int yes = 0;
	vi perm(T);
	for (int i = 0; i < T; i++) perm[i] = i;
	random_shuffle(perm.begin(), perm.end());
	for (int i = 0; i < T; i++) {
		freopen("/dev/tty", "w", stdout);
		printf("%d\n", CK(i));
	}
}
signed main() {
	srand(time(NULL));
	init();
	ANN();
	return 0;
}
