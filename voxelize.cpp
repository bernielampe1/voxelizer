// voxelize.cpp
// C++17 reference implementation of Huang et al. (1998)
// "An Accurate Method for Voxelizing Polygon Meshes":
// vertex-sphere / edge-cylinder / face-slab tests + optional solid fill.
// Writes BINVOX v1.0 (RLE; y-fastest ordering).
//
// Build : make            (no OpenMP)
//         make USE_OMP=1  (try OpenMP)
// Usage : ./voxelize -i mesh.obj -o out.binvox -n 256 [--solid] [--conn 6|18|26] [--threads N]
//
// Notes:
// - Minimal OBJ reader (v,f). Triangulates n-gons via fan.
// - Mesh normalized to [0,1]^3; BINVOX header stores translate=min, scale=max_extent.
// - If OpenMP is enabled, parallel writes are via std::atomic<uint8_t>.
//
// License: MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
  #include <omp.h>
#endif

struct Vec3 {
    double x=0, y=0, z=0;
    Vec3() = default;
    Vec3(double X,double Y,double Z):x(X),y(Y),z(Z){}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
};

inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3   cross(const Vec3& a, const Vec3& b){
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double norm2(const Vec3& v){ return dot(v,v); }
inline double norm (const Vec3& v){ return std::sqrt(norm2(v)); }

struct Tri { int v0, v1, v2; };
struct Mesh { std::vector<Vec3> V; std::vector<Tri> T; };

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size()
        && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// Minimal OBJ loader: supports 'v' and 'f' (triangulates n-gons via fan).
bool loadOBJ(const std::string& path, Mesh& mesh) {
    std::ifstream in(path);
    if(!in) { std::cerr << "Failed to open OBJ: " << path << "\n"; return false; }

    std::vector<Vec3> verts;
    std::vector<Tri>  tris;
    std::string line;

    auto parse_index = [] (const std::string& token) -> int {
        // token like "i", "i/j/k"; negative allowed; only first field used.
        if(token.empty()) return 0;
        std::stringstream ss(token);
        int idx = 0;
        ss >> idx;
        return idx; // keep 1-based for now; caller adjusts
    };

    while(std::getline(in, line)) {
        if(line.empty() || line[0] == '#') continue;
        std::stringstream ls(line);
        std::string tag; ls >> tag;
        if(tag == "v") {
            double x,y,z; ls >> x >> y >> z; verts.emplace_back(x,y,z);
        } else if(tag == "f") {
            std::vector<int> idx;
            std::string tok;
            while(ls >> tok) {
                size_t s = tok.find('/');
                std::string t = (s==std::string::npos) ? tok : tok.substr(0,s);
                int id1 = parse_index(t);
                if(id1 == 0) continue;
                // OBJ indexing: negative counts from end, positive is 1-based
                int id0 = (id1 < 0) ? (int(verts.size()) + 1 + id1) : id1;
                idx.push_back(id0 - 1); // to 0-based
            }
            if(idx.size() >= 3) {
                for(size_t k=1; k+1<idx.size(); ++k)
                    tris.push_back({idx[0], idx[k], idx[k+1]});
            }
        }
    }
    if(verts.empty() || tris.empty()){
        std::cerr << "OBJ contains no usable vertices/triangles.\n";
        return false;
    }
    mesh.V = std::move(verts);
    mesh.T = std::move(tris);
    return true;
}

struct Bounds {
    Vec3 min{  std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity() };
    Vec3 max{ -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity() };
};

Bounds computeBounds(const std::vector<Vec3>& V){
    Bounds b;
    for(const auto& p: V){
        b.min.x = std::min(b.min.x, p.x);
        b.min.y = std::min(b.min.y, p.y);
        b.min.z = std::min(b.min.z, p.z);
        b.max.x = std::max(b.max.x, p.x);
        b.max.y = std::max(b.max.y, p.y);
        b.max.z = std::max(b.max.z, p.z);
    }
    return b;
}

// Map mesh to unit cube [0,1]^3; return original translate+scale for BINVOX header.
// BINVOX expects: world = translate + scale * unit.
struct NormXform { Vec3 translate; double scale = 1.0; };

NormXform normalizeToUnit(Mesh& mesh){
    Bounds b = computeBounds(mesh.V);
    Vec3 ext = b.max - b.min;
    double maxe = std::max({ext.x, ext.y, ext.z});
    double s = (maxe > 0) ? (1.0 / maxe) : 1.0; // scale to unit
    for(auto& p : mesh.V){
        p.x = (p.x - b.min.x) * s;
        p.y = (p.y - b.min.y) * s;
        p.z = (p.z - b.min.z) * s;
    }
    // Header should store inverse mapping: world = unit * (1/s) + b.min
    return { /*translate=*/ b.min, /*scale=*/ (1.0 / s) };
}

inline bool pointInTriProjected(const Vec3& P, const Vec3& A, const Vec3& B, const Vec3& C, const Vec3& N){
    // Edge-function test against triangle normal; accepts boundary.
    Vec3 e0 = B - A, e1 = C - B, e2 = A - C;
    double s0 = dot(cross(e0, P - A), N);
    double s1 = dot(cross(e1, P - B), N);
    double s2 = dot(cross(e2, P - C), N);
    const double eps = 1e-12;
    bool pos = (s0 >= -eps) && (s1 >= -eps) && (s2 >= -eps);
    bool neg = (s0 <= +eps) && (s1 <= +eps) && (s2 <= +eps);
    return pos || neg;
}

inline double distPointSeg(const Vec3& P, const Vec3& A, const Vec3& B){
    Vec3 AB = B - A;
    double l2 = norm2(AB);
    if(l2 == 0.0) return norm(P - A);
    double t = dot(P - A, AB) / l2;
    t = std::max(0.0, std::min(1.0, t));
    Vec3 Q = A + AB * t;
    return norm(P - Q);
}

bool write_binvox(const std::string& path, int N, const Vec3& translate, double scale,
                  const std::vector<uint8_t>& vox)
{
    std::ofstream out(path, std::ios::binary);
    if(!out){ std::cerr << "Failed to open output: " << path << "\n"; return false; }

    out << "#binvox 1\n";
    out << "dim " << N << " " << N << " " << N << "\n";
    out << "translate " << translate.x << " " << translate.y << " " << translate.z << "\n";
    out << "scale " << scale << "\n";
    out << "data\n";

    auto idx = [N] (int x, int y, int z) -> size_t {
        // y fastest, then z, then x (binvox convention)
        return static_cast<size_t>(x)*N*N + static_cast<size_t>(z)*N + static_cast<size_t>(y);
    };

    uint8_t current = 0;
    uint8_t count   = 0;
    auto flush = [&out] (uint8_t val, uint8_t cnt){
        out.put(static_cast<char>(val));
        out.put(static_cast<char>(cnt));
    };

    for(int x=0;x<N;++x){
        for(int z=0;z<N;++z){
            for(int y=0;y<N;++y){
                uint8_t v = vox[idx(x,y,z)] ? 1 : 0;
                if(count == 0){ current = v; count = 1; }
                else if(v == current && count < 255){ ++count; }
                else { flush(current, count); current = v; count = 1; }
            }
        }
    }
    if(count) flush(current, count);
    return true;
}

struct ConnTuning {
    double k_face=1.0, k_edge=1.0, k_vert=1.0;
    static ConnTuning for_conn(int conn){
        if(conn == 6)   return {0.90, 0.95, 0.95}; // thinner
        if(conn == 26)  return {1.10, 1.05, 1.05}; // thicker
        return {1.00, 1.00, 1.00};                 // default (18)
    }
};

struct Voxelizer {
    const Mesh& mesh;
    int N;
    bool solid;
    ConnTuning tune;
    double vox; // edge length (unit cube -> 1/N)

    Voxelizer(const Mesh& m, int n, bool makeSolid, ConnTuning t)
    : mesh(m), N(n), solid(makeSolid), tune(t), vox(1.0 / double(n)) {}

    std::vector<uint8_t> run() const {
        const size_t total = static_cast<size_t>(N)*N*N;

        // Use atomics when OpenMP is enabled; relaxed ordering is enough.
        std::vector<std::atomic<uint8_t>> grid_atomic(total);
        for(auto& a : grid_atomic) a.store(0, std::memory_order_relaxed);

        const double r_face  = tune.k_face * 0.5 * vox;
        const double r_edge  = tune.k_edge * std::sqrt(2.0)/2.0 * vox;
        const double r_vert  = tune.k_vert * std::sqrt(3.0)/2.0 * vox;
        const double r_max   = std::max({r_face, r_edge, r_vert});
        const double r_face2 = r_face*r_face;
        const double r_edge2 = r_edge*r_edge;
        const double r_vert2 = r_vert*r_vert;

        auto centerOf = [this] (int i, int j, int k) -> Vec3 {
            return { (i + 0.5) * vox, (j + 0.5) * vox, (k + 0.5) * vox };
        };
        auto idx = [this] (int x, int y, int z) -> size_t {
            return static_cast<size_t>(x)*N*N + static_cast<size_t>(z)*N + static_cast<size_t>(y);
        };

        const size_t Tn = mesh.T.size();

        #pragma omp parallel for schedule(dynamic) if(Tn>32)
        for (ptrdiff_t ti = 0; ti < (ptrdiff_t)Tn; ++ti) {
            const Tri& t = mesh.T[(size_t)ti];
            const Vec3 A = mesh.V[(size_t)t.v0];
            const Vec3 B = mesh.V[(size_t)t.v1];
            const Vec3 C = mesh.V[(size_t)t.v2];

            Vec3 Nn = cross(B-A, C-A);
            double area2 = norm(Nn);
            if(area2 == 0.0) continue;
            Vec3 n = Nn / area2; // unit
            double d = -dot(n, A);

            Vec3 triMin{ std::min({A.x,B.x,C.x}), std::min({A.y,B.y,C.y}), std::min({A.z,B.z,C.z}) };
            Vec3 triMax{ std::max({A.x,B.x,C.x}), std::max({A.y,B.y,C.y}), std::max({A.z,B.z,C.z}) };

            int ix0 = std::max(0,   (int)std::floor((triMin.x - r_max)/vox));
            int iy0 = std::max(0,   (int)std::floor((triMin.y - r_max)/vox));
            int iz0 = std::max(0,   (int)std::floor((triMin.z - r_max)/vox));
            int ix1 = std::min(N-1, (int)std::ceil ((triMax.x + r_max)/vox));
            int iy1 = std::min(N-1, (int)std::ceil ((triMax.y + r_max)/vox));
            int iz1 = std::min(N-1, (int)std::ceil ((triMax.z + r_max)/vox));

            const std::array<std::pair<Vec3,Vec3>,3> edges = {{{A,B},{B,C},{C,A}}};

            for(int i = ix0; i <= ix1; ++i){
                for(int k = iz0; k <= iz1; ++k){
                    for(int j = iy0; j <= iy1; ++j){
                        Vec3 Cc = centerOf(i,j,k);

                        // Vertex tests
                        double dA2 = norm2(Cc - A);
                        if(dA2 <= r_vert2){ grid_atomic[idx(i,j,k)].store(1, std::memory_order_relaxed); continue; }
                        double dB2 = norm2(Cc - B);
                        if(dB2 <= r_vert2){ grid_atomic[idx(i,j,k)].store(1, std::memory_order_relaxed); continue; }
                        double dC2 = norm2(Cc - C);
                        if(dC2 <= r_vert2){ grid_atomic[idx(i,j,k)].store(1, std::memory_order_relaxed); continue; }

                        // Edge-cylinder tests
                        bool hitEdge = false;
                        for(const auto& e : edges){
                            double de = distPointSeg(Cc, e.first, e.second);
                            if(de*de <= r_edge2){ hitEdge = true; break; }
                        }
                        if(hitEdge){ grid_atomic[idx(i,j,k)].store(1, std::memory_order_relaxed); continue; }

                        // Face slab test
                        double sd = dot(n, Cc) + d;
                        if(std::abs(sd) <= r_face){
                            Vec3 P = Cc - n*sd; // projection
                            if(pointInTriProjected(P, A, B, C, n)){
                                grid_atomic[idx(i,j,k)].store(1, std::memory_order_relaxed); continue;
                            }
                        }
                    }
                }
            }
        }

        // Convert to plain vector for optional solid fill and I/O.
        std::vector<uint8_t> grid(total, 0);
        for(size_t i=0;i<total;++i) grid[i] = grid_atomic[i].load(std::memory_order_relaxed);

        if(!solid) return grid;

        // --- Solid fill: 6-connected flood from boundary through empty voxels ---
        std::vector<uint8_t> visited(total, 0);
        std::queue<std::array<int,3>> q;
        auto push_if = [&] (int x, int y, int z){
            if(x<0||x>=N||y<0||y>=N||z<0||z>=N) return;
            size_t id = idx(x,y,z);
            if(visited[id] || grid[id]) return;
            visited[id] = 1; q.push({x,y,z});
        };

        for(int x=0;x<N;++x) for(int z=0;z<N;++z){ push_if(x,0,z); push_if(x,N-1,z); }
        for(int y=0;y<N;++y) for(int z=0;z<N;++z){ push_if(0,y,z); push_if(N-1,y,z); }
        for(int x=0;x<N;++x) for(int y=0;y<N;++y){ push_if(x,y,0); push_if(x,y,N-1); }

        const int dx[6]={1,-1,0,0,0,0};
        const int dy[6]={0,0,1,-1,0,0};
        const int dz[6]={0,0,0,0,1,-1};

        while(!q.empty()){
            auto a = q.front(); q.pop();
            for(int t=0;t<6;++t){
                int nx=a[0]+dx[t], ny=a[1]+dy[t], nz=a[2]+dz[t];
                if(nx<0||nx>=N||ny<0||ny>=N||nz<0||nz>=N) continue;
                size_t id = idx(nx,ny,nz);
                if(!visited[id] && !grid[id]){ visited[id]=1; q.push({nx,ny,nz}); }
            }
        }

        for(int x=0;x<N;++x)
            for(int z=0;z<N;++z)
                for(int y=0;y<N;++y){
                    size_t id = idx(x,y,z);
                    if(!visited[id]) grid[id] = 1; // interior (or already surface)
                }

        return grid;
    }
};

struct Args {
    std::string in, out;
    int N = 256;
    bool solid = false;
    int conn = 18;
    int threads = -1;
};

bool parseArgs(int argc, char** argv, Args& a){
    auto need = [argc] (int i){ return i+1 < argc; };

    for(int i=1;i<argc;++i){
        std::string s(argv[i]);
        if((s=="-i"||s=="--input") && need(i))       a.in = argv[++i];
        else if((s=="-o"||s=="--output") && need(i)) a.out = argv[++i];
        else if((s=="-n"||s=="--dim") && need(i))    a.N = std::max(4, std::stoi(argv[++i]));
        else if(s=="--solid")                        a.solid = true;
        else if(s=="--conn" && need(i))              { a.conn = std::stoi(argv[++i]); if(a.conn!=6 && a.conn!=18 && a.conn!=26) a.conn=18; }
        else if(s=="--threads" && need(i))           a.threads = std::stoi(argv[++i]);
        else if(s=="-h"||s=="--help"){
            std::cout << "Usage: voxelize -i mesh.obj -o out.binvox -n 256 [--solid] [--conn 6|18|26] [--threads N]\n";
            return false;
        }
    }
    if(a.in.empty() || a.out.empty()){
        std::cerr << "Error: input/output required.\n";
        return false;
    }
    if(!ends_with(a.out, ".binvox")){
        std::cerr << "Warning: output does not end with .binvox; proceeding anyway.\n";
    }
    return true;
}

int main(int argc, char** argv){
    Args a;
    if(!parseArgs(argc, argv, a)) return 1;

#if defined(_OPENMP)
    if(a.threads > 0) omp_set_num_threads(a.threads);
#endif

    Mesh mesh;
    if(!loadOBJ(a.in, mesh)) return 2;
    auto xform = normalizeToUnit(mesh);

    ConnTuning tune = ConnTuning::for_conn(a.conn);
    Voxelizer vox(mesh, a.N, a.solid, tune);
    auto grid = vox.run();

    if(!write_binvox(a.out, a.N, xform.translate, xform.scale, grid)){
        std::cerr << "Failed to write output.\n";
        return 3;
    }
    std::cout << "Wrote " << a.out << " (" << a.N << "^3)\n";
    return 0;
}
