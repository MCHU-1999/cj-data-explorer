// plane_detect.cpp
//
// Detect planes in a point cloud (PLY) with CGAL and write them as polygons:
// each plane's points are fitted with a least-squares plane, projected onto
// it, and outlined by their 2D convex hull. One polygon face per plane.
//
// Output PLY (binary):
//   vertex: x y z (double)
//   face:   vertex_indices (list int int), red green blue (uchar), plane_id (int)
//
// Two detectors are available:
//   --method ransac  CGAL Efficient RANSAC (default; fast on large clouds)
//   --method rg      CGAL region growing (connected planar regions, k-NN based)
//
// Usage:
//   plane_detect input.ply output.ply [options]
//
// Options (units are the units of your coordinates, usually metres):
//   --method ransac|rg     detector (default: ransac)
//   --dist D               max point-to-plane distance          (default 0.10)
//   --angle A              max normal deviation, degrees         (default 20)
//   --min-points N         min points per plane                  (default 200)
//   --cluster-eps C        RANSAC only: max gap between points of one
//                          plane instance, ~2x point spacing     (default 0.50)
//   --probability P        RANSAC only: chance to miss a shape   (default 0.01)
//   --k K                  neighbours for normals / region growing (default 12)
//   --voxel V              grid-simplify to cell size V first (0 = off, default)
//   --recompute-normals    ignore normals stored in the input PLY
//
// Requires CGAL >= 5.5, Eigen3; TBB optional (parallel normal estimation).

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/convex_hull_2.h>

#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Point_set.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Kernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT        = Kernel::FT;
using Point     = Kernel::Point_3;
using Point_2   = Kernel::Point_2;
using Vector    = Kernel::Vector_3;
using Plane_3   = Kernel::Plane_3;
using Point_set = CGAL::Point_set_3<Point, Vector>;
using Index     = Point_set::Index;

// Efficient RANSAC
using Ransac_traits = CGAL::Shape_detection::Efficient_RANSAC_traits<
    Kernel, Point_set, Point_set::Point_map, Point_set::Vector_map>;
using Efficient_ransac = CGAL::Shape_detection::Efficient_RANSAC<Ransac_traits>;
using Ransac_plane     = CGAL::Shape_detection::Plane<Ransac_traits>;

// Region growing
namespace PS = CGAL::Shape_detection::Point_set;
using Neighbor_query = PS::K_neighbor_query_for_point_set<Point_set>;
using Region_type    = PS::Least_squares_plane_fit_region_for_point_set<Point_set>;
using Sorting        = PS::Least_squares_plane_fit_sorting_for_point_set<Point_set, Neighbor_query>;
using Region_growing = CGAL::Shape_detection::Region_growing<Neighbor_query, Region_type>;

struct Options {
  std::string input, output;
  std::string method = "ransac";
  double dist = 0.10;
  double angle_deg = 15.0;
  std::size_t min_points = 200;
  double cluster_eps = 0.50;
  double probability = 0.01;
  unsigned int k = 12;
  double voxel = 0.0;
  bool recompute_normals = false;
};

// ---------------------------------------------------------------------------

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " input.ply output.ply [options]\n"
            << "  --method ransac|rg   (default ransac)\n"
            << "  --dist D             max point-to-plane distance (0.10)\n"
            << "  --angle A            max normal deviation in degrees (15)\n"
            << "  --min-points N       min points per plane (200)\n"
            << "  --cluster-eps C      RANSAC connectivity gap (0.50)\n"
            << "  --probability P      RANSAC miss probability (0.01)\n"
            << "  --k K                neighbours for normals / rg (12)\n"
            << "  --voxel V            grid-simplify cell size, 0 = off (0)\n"
            << "  --recompute-normals  ignore normals in the input\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
  if (argc < 3) return false;
  o.input = argv[1];
  o.output = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
      return argv[++i];
    };
    if      (a == "--method")            o.method = next();
    else if (a == "--dist")              o.dist = std::stod(next());
    else if (a == "--angle")             o.angle_deg = std::stod(next());
    else if (a == "--min-points")        o.min_points = std::stoul(next());
    else if (a == "--cluster-eps")       o.cluster_eps = std::stod(next());
    else if (a == "--probability")       o.probability = std::stod(next());
    else if (a == "--k")                 o.k = static_cast<unsigned int>(std::stoul(next()));
    else if (a == "--voxel")             o.voxel = std::stod(next());
    else if (a == "--recompute-normals") o.recompute_normals = true;
    else throw std::runtime_error("unknown option " + a);
  }
  if (o.method != "ransac" && o.method != "rg")
    throw std::runtime_error("--method must be 'ransac' or 'rg'");
  if (o.dist <= 0 || o.angle_deg <= 0 || o.angle_deg >= 90 || o.k < 3)
    throw std::runtime_error("invalid --dist / --angle / --k");
  return true;
}

// Deterministic, well-spread colours: golden-ratio hue + varied sat/value.
static std::array<unsigned char, 3> plane_color(std::size_t i) {
  const double h = std::fmod(static_cast<double>(i) * 0.618033988749895, 1.0) * 6.0;
  const double s = 0.55 + 0.45 * static_cast<double>((i * 7) % 5) / 4.0;
  const double v = 0.70 + 0.30 * static_cast<double>((i * 3) % 4) / 3.0;
  const int    sector = static_cast<int>(h) % 6;
  const double f = h - std::floor(h);
  const double p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  double r = 0, g = 0, b = 0;
  switch (sector) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return {static_cast<unsigned char>(std::lround(r * 255)),
          static_cast<unsigned char>(std::lround(g * 255)),
          static_cast<unsigned char>(std::lround(b * 255))};
}

class Timer {
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
 public:
  double sec() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }
};

// ---------------------------------------------------------------------------
// Detectors: each returns one vector of point indices per detected plane.

static std::vector<std::vector<Index>> detect_ransac(Point_set& points, const Options& o) {
  Efficient_ransac ransac;
  ransac.set_input(points, points.point_map(), points.normal_map());
  ransac.add_shape_factory<Ransac_plane>();

  Efficient_ransac::Parameters p;
  p.probability      = o.probability;
  p.min_points       = o.min_points;
  p.epsilon          = o.dist;
  p.cluster_epsilon  = o.cluster_eps;
  p.normal_threshold = std::cos(o.angle_deg * CGAL_PI / 180.0);

  ransac.preprocess();
  ransac.detect(p);

  std::vector<std::vector<Index>> planes;
  planes.reserve(ransac.shapes().size());
  for (const auto& shape : ransac.shapes()) {
    const std::vector<std::size_t>& idx = shape->indices_of_assigned_points();
    std::vector<Index> plane;
    plane.reserve(idx.size());
    for (std::size_t j : idx) plane.push_back(*(points.begin() + j));
    planes.push_back(std::move(plane));
  }
  return planes;
}

static std::vector<std::vector<Index>> detect_region_growing(Point_set& points, const Options& o) {
  Neighbor_query neighbor_query =
      PS::make_k_neighbor_query(points, CGAL::parameters::k_neighbors(o.k));

  Sorting sorting = PS::make_least_squares_plane_fit_sorting(points, neighbor_query);
  sorting.sort();

  Region_type region_type = PS::make_least_squares_plane_fit_region(
      points, CGAL::parameters::maximum_distance(FT(o.dist))
                  .maximum_angle(FT(o.angle_deg))
                  .minimum_region_size(o.min_points));

  Region_growing region_growing(points, sorting.ordered(), neighbor_query, region_type);

  std::vector<Region_growing::Primitive_and_region> regions;
  region_growing.detect(std::back_inserter(regions));

  std::vector<std::vector<Index>> planes;
  planes.reserve(regions.size());
  for (const auto& pr : regions) {
    std::vector<Index> plane(pr.second.begin(), pr.second.end());
    planes.push_back(std::move(plane));
  }
  return planes;
}

// ---------------------------------------------------------------------------
// Plane points -> flat convex polygon (vertices in CCW order around the normal)

struct Polygon3 {
  std::vector<Point> ring;
};

static Polygon3 plane_polygon(const Point_set& points, const std::vector<Index>& idx) {
  std::vector<Point> pts;
  pts.reserve(idx.size());
  for (Index i : idx) pts.push_back(points.point(i));

  Plane_3 plane;
  Point centroid;
  CGAL::linear_least_squares_fitting_3(pts.begin(), pts.end(), plane, centroid,
                                       CGAL::Dimension_tag<0>());

  // Orthonormal frame (u, v, n), right-handed; normal points up when possible.
  Vector n = plane.orthogonal_vector();
  n = n / std::sqrt(CGAL::to_double(n.squared_length()));
  if (n.z() < 0) n = -n;
  Vector u = plane.base1();
  u = u / std::sqrt(CGAL::to_double(u.squared_length()));
  Vector v = CGAL::cross_product(n, u);

  std::vector<Point_2> pts2;
  pts2.reserve(pts.size());
  for (const Point& p : pts) {
    const Vector d = p - centroid;
    pts2.emplace_back(d * u, d * v);
  }

  std::vector<Point_2> hull;
  CGAL::convex_hull_2(pts2.begin(), pts2.end(), std::back_inserter(hull));  // CCW

  Polygon3 poly;
  poly.ring.reserve(hull.size());
  for (const Point_2& q : hull) poly.ring.push_back(centroid + q.x() * u + q.y() * v);
  return poly;
}

template <typename T>
static void write_bin(std::ostream& os, T v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Binary little-endian PLY, one polygon face per plane.
static bool write_polygons_ply(const std::string& path, const std::vector<Polygon3>& polys,
                               const std::vector<int>& ids) {
  std::size_t n_verts = 0;
  for (const auto& p : polys) n_verts += p.ring.size();

  std::ofstream os(path, std::ios::binary);
  if (!os) return false;
  os << "ply\nformat binary_little_endian 1.0\n"
     << "element vertex " << n_verts << "\n"
     << "property double x\nproperty double y\nproperty double z\n"
     << "element face " << polys.size() << "\n"
     << "property list int int vertex_indices\n"
     << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
     << "property int plane_id\n"
     << "end_header\n";

  for (const auto& p : polys)
    for (const Point& q : p.ring) {
      write_bin(os, CGAL::to_double(q.x()));
      write_bin(os, CGAL::to_double(q.y()));
      write_bin(os, CGAL::to_double(q.z()));
    }

  int base = 0;
  for (std::size_t i = 0; i < polys.size(); ++i) {
    const int k = static_cast<int>(polys[i].ring.size());
    write_bin(os, k);
    for (int j = 0; j < k; ++j) write_bin(os, base + j);
    const auto c = plane_color(static_cast<std::size_t>(ids[i]));
    write_bin(os, c[0]);
    write_bin(os, c[1]);
    write_bin(os, c[2]);
    write_bin(os, ids[i]);
    base += k;
  }
  return static_cast<bool>(os);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Options o;
  try {
    if (!parse_args(argc, argv, o)) { usage(argv[0]); return EXIT_FAILURE; }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  Timer total;

  // 1. Read ------------------------------------------------------------------
  Point_set points;
  {
    Timer t;
    std::cout << "Reading " << o.input << " ..." << std::flush;
    if (!CGAL::IO::read_point_set(o.input, points)) {
      std::cerr << "\nError: cannot read " << o.input << "\n";
      return EXIT_FAILURE;
    }
    std::cout << " " << points.size() << " points (" << t.sec() << " s)\n";
  }
  if (points.empty()) { std::cerr << "Error: empty point cloud\n"; return EXIT_FAILURE; }

  // 2. Optional voxel downsampling --------------------------------------------
  if (o.voxel > 0) {
    Timer t;
    std::cout << "Grid simplification (cell " << o.voxel << ") ..." << std::flush;
    points.remove(CGAL::grid_simplify_point_set(points, o.voxel, points.parameters()),
                  points.end());
    points.collect_garbage();
    std::cout << " " << points.size() << " points left (" << t.sec() << " s)\n";
  }

  // 3. Normals ----------------------------------------------------------------
  if (!points.has_normal_map() || o.recompute_normals) {
    Timer t;
    std::cout << "Estimating normals (PCA, k=" << o.k << ") ..." << std::flush;
    points.add_normal_map();
    CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(
        points, o.k, points.parameters());
    std::cout << " done (" << t.sec() << " s)\n";
  } else {
    // Input normals are not guaranteed to be unit length (MicMac C3DC writes
    // |n| = 0.1), but RANSAC's normal_threshold assumes unit normals.
    std::size_t n_zero = 0;
    for (Index i : points) {
      const Vector n = points.normal(i);
      const double len = std::sqrt(CGAL::to_double(n.squared_length()));
      if (len > 1e-12) points.normal(i) = n / len;
      else ++n_zero;
    }
    std::cout << "Using normals from input file (normalised";
    if (n_zero) std::cout << ", " << n_zero << " zero-length";
    std::cout << ")\n";
  }

  // 4. Plane detection ---------------------------------------------------------
  std::vector<std::vector<Index>> planes;
  {
    Timer t;
    std::cout << "Detecting planes (" << (o.method == "ransac" ? "Efficient RANSAC" : "region growing")
              << ", dist=" << o.dist << ", angle=" << o.angle_deg
              << " deg, min_points=" << o.min_points;
    if (o.method == "ransac") std::cout << ", cluster_eps=" << o.cluster_eps;
    std::cout << ") ..." << std::flush;

    planes = (o.method == "ransac") ? detect_ransac(points, o)
                                    : detect_region_growing(points, o);
    std::cout << " " << planes.size() << " planes (" << t.sec() << " s)\n";
  }

  // 5. Fit each plane and outline it with its convex hull ------------------------
  std::vector<Polygon3> polys;
  std::vector<int> ids;
  std::size_t n_kept = 0;
  {
    Timer t;
    std::cout << "Building convex-hull polygons ..." << std::flush;
    polys.reserve(planes.size());
    ids.reserve(planes.size());
    for (std::size_t i = 0; i < planes.size(); ++i) {
      n_kept += planes[i].size();
      Polygon3 poly = plane_polygon(points, planes[i]);
      if (poly.ring.size() < 3) continue;  // degenerate (collinear) plane
      polys.push_back(std::move(poly));
      ids.push_back(static_cast<int>(i));
    }
    std::cout << " " << polys.size() << " polygons (" << t.sec() << " s)\n";
  }

  const double kept_pct = 100.0 * static_cast<double>(n_kept) / static_cast<double>(points.size());
  std::cout << n_kept << " / " << points.size() << " points on planes (" << kept_pct << " %)\n";

  // 6. Write (binary PLY) -------------------------------------------------------
  {
    Timer t;
    if (!write_polygons_ply(o.output, polys, ids)) {
      std::cerr << "Error: failed writing " << o.output << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "Wrote " << o.output << " (" << t.sec() << " s)\n";
  }

  std::cout << "Total time: " << total.sec() << " s\n";
  return EXIT_SUCCESS;
}