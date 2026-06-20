#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double EPS = 1e-12;
constexpr double PI = 3.141592653589793238462643383279502884;

bool is_finite(double value) { return std::isfinite(value); }

std::vector<double> to_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto view = arr.unchecked<1>();
    std::vector<double> out(view.shape(0));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) out[static_cast<size_t>(i)] = view(i);
    return out;
}

std::vector<double> object_vec(py::object obj) {
    if (obj.is_none()) return {};
    return to_vec(py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(obj));
}

py::array_t<double> to_array(const std::vector<double>& values) {
    py::array_t<double> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for (size_t i = 0; i < values.size(); ++i) view(static_cast<py::ssize_t>(i)) = values[i];
    return out;
}

std::vector<double> finite_vec(const std::vector<double>& values) {
    std::vector<double> out;
    out.reserve(values.size());
    for (double value : values) {
        if (is_finite(value)) out.push_back(value);
    }
    return out;
}

int default_min_periods(int window, int min_periods) {
    if (min_periods > 0) return min_periods;
    return std::min(std::max(3, window / 4), window);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return NaN;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double variance_sample(const std::vector<double>& values) {
    if (values.size() < 2) return NaN;
    double m = mean(values);
    double ss = 0.0;
    for (double value : values) ss += (value - m) * (value - m);
    return ss / static_cast<double>(values.size() - 1);
}

double variance_population(const std::vector<double>& values) {
    if (values.empty()) return NaN;
    double m = mean(values);
    double ss = 0.0;
    for (double value : values) ss += (value - m) * (value - m);
    return ss / static_cast<double>(values.size());
}

double chi2_sf_approx(double statistic, int dof) {
    if (!is_finite(statistic) || dof <= 0) return NaN;
    double z = (std::pow(statistic / static_cast<double>(dof), 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) /
               std::sqrt(2.0 / (9.0 * dof));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

std::vector<double> rolling_apply(const std::vector<double>& values, int window, int min_periods, const std::string& stat) {
    std::vector<double> out(values.size(), NaN);
    int periods = default_min_periods(window, min_periods);
    for (size_t i = 0; i < values.size(); ++i) {
        size_t start = (i + 1 > static_cast<size_t>(window)) ? i + 1 - static_cast<size_t>(window) : 0;
        std::vector<double> clean;
        for (size_t j = start; j <= i; ++j) {
            if (is_finite(values[j])) clean.push_back(values[j]);
        }
        if (clean.size() < static_cast<size_t>(periods)) continue;
        if (stat == "mean") {
            out[i] = mean(clean);
        } else if (stat == "variance") {
            out[i] = variance_sample(clean);
        } else if (stat == "std") {
            double v = variance_sample(clean);
            out[i] = is_finite(v) && v >= 0.0 ? std::sqrt(v) : NaN;
        } else if (stat == "skew") {
            double m = mean(clean);
            double s2 = variance_population(clean);
            double s = std::sqrt(std::max(s2, 0.0));
            if (s <= 0.0) {
                out[i] = 0.0;
            } else {
                double third = 0.0;
                for (double value : clean) third += std::pow((value - m) / s, 3.0);
                out[i] = third / static_cast<double>(clean.size());
            }
        } else if (stat == "kurt") {
            double m = mean(clean);
            double s2 = variance_population(clean);
            double s = std::sqrt(std::max(s2, 0.0));
            if (s <= 0.0) {
                out[i] = 0.0;
            } else {
                double fourth = 0.0;
                for (double value : clean) fourth += std::pow((value - m) / s, 4.0);
                out[i] = fourth / static_cast<double>(clean.size()) - 3.0;
            }
        }
    }
    return out;
}

std::vector<double> log_price(py::array_t<double, py::array::c_style | py::array::forcecast> prices) {
    auto values = to_vec(prices);
    std::vector<double> out(values.size(), NaN);
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] > 0.0 && is_finite(values[i])) out[i] = std::log(values[i]);
    }
    return out;
}

std::vector<double> log_returns(py::array_t<double, py::array::c_style | py::array::forcecast> prices) {
    auto values = to_vec(prices);
    std::vector<double> out(values.size(), NaN);
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > 0.0 && values[i - 1] > 0.0 && is_finite(values[i]) && is_finite(values[i - 1])) {
            out[i] = std::log(values[i] / values[i - 1]);
        }
    }
    return out;
}

std::vector<double> difference(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto values = to_vec(arr);
    std::vector<double> out(values.size(), NaN);
    for (size_t i = 1; i < values.size(); ++i) out[i] = values[i] - values[i - 1];
    return out;
}

std::vector<double> seasonal_difference(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int period) {
    auto values = to_vec(arr);
    int p = std::max(1, period);
    std::vector<double> out(values.size(), NaN);
    for (size_t i = static_cast<size_t>(p); i < values.size(); ++i) out[i] = values[i] - values[i - static_cast<size_t>(p)];
    return out;
}

std::vector<double> ewma_variance(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int span) {
    auto values = to_vec(arr);
    std::vector<double> out(values.size(), NaN);
    if (values.empty()) return out;
    double alpha = 2.0 / (std::max(1, span) + 1.0);
    double variance = NaN;
    int seen = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!is_finite(values[i])) continue;
        double sq = values[i] * values[i];
        variance = is_finite(variance) ? alpha * sq + (1.0 - alpha) * variance : sq;
        ++seen;
        if (seen >= 3) out[i] = variance;
    }
    return out;
}

std::vector<double> range_based_volatility(py::array_t<double, py::array::c_style | py::array::forcecast> high,
                                           py::array_t<double, py::array::c_style | py::array::forcecast> low) {
    auto h = to_vec(high);
    auto l = to_vec(low);
    size_t n = std::min(h.size(), l.size());
    std::vector<double> out(n, NaN);
    for (size_t i = 0; i < n; ++i) {
        if (h[i] > 0.0 && l[i] > 0.0 && is_finite(h[i]) && is_finite(l[i])) {
            double hl = std::log(h[i] / l[i]);
            out[i] = std::sqrt((hl * hl) / (4.0 * std::log(2.0)));
        }
    }
    return out;
}

std::vector<double> rolling_pair(py::array_t<double, py::array::c_style | py::array::forcecast> left,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> right,
                                 int window,
                                 int min_periods,
                                 bool corr) {
    auto a = to_vec(left);
    auto b = to_vec(right);
    size_t n = std::min(a.size(), b.size());
    std::vector<double> out(n, NaN);
    int periods = default_min_periods(window, min_periods);
    for (size_t i = 0; i < n; ++i) {
        size_t start = (i + 1 > static_cast<size_t>(window)) ? i + 1 - static_cast<size_t>(window) : 0;
        std::vector<double> x, y;
        for (size_t j = start; j <= i; ++j) {
            if (is_finite(a[j]) && is_finite(b[j])) {
                x.push_back(a[j]);
                y.push_back(b[j]);
            }
        }
        if (x.size() < static_cast<size_t>(periods) || x.size() < 2) continue;
        double mx = mean(x), my = mean(y);
        double cov = 0.0, vx = 0.0, vy = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            cov += (x[j] - mx) * (y[j] - my);
            vx += (x[j] - mx) * (x[j] - mx);
            vy += (y[j] - my) * (y[j] - my);
        }
        if (corr) {
            out[i] = (vx <= 0.0 || vy <= 0.0) ? NaN : cov / std::sqrt(vx * vy);
        } else {
            out[i] = cov / static_cast<double>(x.size() - 1);
        }
    }
    return out;
}

std::vector<double> acf(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int max_lag) {
    auto values = finite_vec(to_vec(arr));
    std::vector<double> out(static_cast<size_t>(max_lag) + 1, NaN);
    if (values.empty()) return out;
    double m = mean(values);
    double denom = 0.0;
    for (double value : values) denom += (value - m) * (value - m);
    out[0] = 1.0;
    for (int lag = 1; lag <= max_lag; ++lag) {
        if (static_cast<size_t>(lag) >= values.size() || denom <= 0.0) continue;
        double num = 0.0;
        for (size_t i = 0; i + static_cast<size_t>(lag) < values.size(); ++i) {
            num += (values[i] - m) * (values[i + static_cast<size_t>(lag)] - m);
        }
        out[static_cast<size_t>(lag)] = num / denom;
    }
    return out;
}

std::vector<double> solve_linear(std::vector<std::vector<double>> a, std::vector<double> b) {
    size_t n = b.size();
    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        for (size_t r = i + 1; r < n; ++r) {
            if (std::abs(a[r][i]) > std::abs(a[pivot][i])) pivot = r;
        }
        if (std::abs(a[pivot][i]) < EPS) return std::vector<double>(n, NaN);
        std::swap(a[i], a[pivot]);
        std::swap(b[i], b[pivot]);
        double div = a[i][i];
        for (size_t c = i; c < n; ++c) a[i][c] /= div;
        b[i] /= div;
        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            double factor = a[r][i];
            for (size_t c = i; c < n; ++c) a[r][c] -= factor * a[i][c];
            b[r] -= factor * b[i];
        }
    }
    return b;
}

std::vector<double> pacf_yule_walker(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int max_lag) {
    auto rho = acf(arr, max_lag);
    std::vector<double> out(static_cast<size_t>(max_lag) + 1, NaN);
    out[0] = 1.0;
    for (int lag = 1; lag <= max_lag; ++lag) {
        std::vector<std::vector<double>> toeplitz(static_cast<size_t>(lag), std::vector<double>(static_cast<size_t>(lag)));
        std::vector<double> r(static_cast<size_t>(lag));
        for (int i = 0; i < lag; ++i) {
            r[static_cast<size_t>(i)] = rho[static_cast<size_t>(i + 1)];
            for (int j = 0; j < lag; ++j) toeplitz[static_cast<size_t>(i)][static_cast<size_t>(j)] = rho[static_cast<size_t>(std::abs(i - j))];
        }
        auto phi = solve_linear(toeplitz, r);
        out[static_cast<size_t>(lag)] = phi.empty() ? NaN : phi.back();
    }
    return out;
}

std::vector<double> rolling_ols(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int window, int min_periods, bool tstat) {
    auto values = to_vec(arr);
    std::vector<double> out(values.size(), NaN);
    int periods = default_min_periods(window, min_periods > 0 ? min_periods : std::min(std::max(4, window / 3), window));
    for (size_t i = 0; i < values.size(); ++i) {
        size_t start = (i + 1 > static_cast<size_t>(window)) ? i + 1 - static_cast<size_t>(window) : 0;
        std::vector<double> y;
        for (size_t j = start; j <= i; ++j) {
            if (is_finite(values[j])) y.push_back(values[j]);
        }
        if (y.size() < static_cast<size_t>(periods) || y.size() < 2) continue;
        double n = static_cast<double>(y.size());
        double xmean = (n - 1.0) / 2.0;
        double ymean = mean(y);
        double denom = 0.0, num = 0.0;
        for (size_t j = 0; j < y.size(); ++j) {
            double x = static_cast<double>(j) - xmean;
            denom += x * x;
            num += x * (y[j] - ymean);
        }
        if (denom <= 0.0) continue;
        double slope = num / denom;
        if (!tstat) {
            out[i] = slope;
        } else {
            if (y.size() < 4) continue;
            double sse = 0.0;
            for (size_t j = 0; j < y.size(); ++j) {
                double x = static_cast<double>(j) - xmean;
                double resid = y[j] - (ymean + slope * x);
                sse += resid * resid;
            }
            double se = std::sqrt((sse / std::max(1.0, n - 2.0)) / denom);
            out[i] = se <= 0.0 ? 0.0 : slope / se;
        }
    }
    return out;
}

double qlike_scalar(double realized, double forecast) {
    return (is_finite(realized) && is_finite(forecast) && forecast > 0.0) ? realized / forecast + std::log(forecast) : NaN;
}

std::vector<double> qlike(py::array_t<double, py::array::c_style | py::array::forcecast> realized,
                          py::array_t<double, py::array::c_style | py::array::forcecast> forecast) {
    auto r = to_vec(realized);
    auto f = to_vec(forecast);
    size_t n = std::min(r.size(), f.size());
    std::vector<double> out(n, NaN);
    for (size_t i = 0; i < n; ++i) out[i] = qlike_scalar(r[i], f[i]);
    return out;
}

double loss_metric(py::array_t<double, py::array::c_style | py::array::forcecast> actual,
                   py::array_t<double, py::array::c_style | py::array::forcecast> forecast,
                   const std::string& metric) {
    auto a = to_vec(actual);
    auto f = to_vec(forecast);
    size_t n = std::min(a.size(), f.size());
    double sum = 0.0;
    int count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!is_finite(a[i]) || !is_finite(f[i])) continue;
        double err = f[i] - a[i];
        if (metric == "rmse") sum += err * err;
        else if (metric == "mae") sum += std::abs(err);
        else sum += err;
        ++count;
    }
    if (count == 0) return NaN;
    double avg = sum / static_cast<double>(count);
    return metric == "rmse" ? std::sqrt(avg) : avg;
}

double corr_metric(py::array_t<double, py::array::c_style | py::array::forcecast> left,
                   py::array_t<double, py::array::c_style | py::array::forcecast> right) {
    auto a = to_vec(left);
    auto b = to_vec(right);
    size_t n = std::min(a.size(), b.size());
    std::vector<double> x, y;
    for (size_t i = 0; i < n; ++i) {
        if (is_finite(a[i]) && is_finite(b[i])) {
            x.push_back(a[i]);
            y.push_back(b[i]);
        }
    }
    if (x.size() < 3) return NaN;
    double mx = mean(x), my = mean(y), cov = 0.0, vx = 0.0, vy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - mx) * (y[i] - my);
        vx += (x[i] - mx) * (x[i] - mx);
        vy += (y[i] - my) * (y[i] - my);
    }
    return (vx <= 0.0 || vy <= 0.0) ? NaN : cov / std::sqrt(vx * vy);
}

py::dict ar_fit(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int p) {
    auto values = finite_vec(to_vec(arr));
    p = std::max(1, p);
    if (values.size() <= static_cast<size_t>(p + 1)) return py::dict("success"_a=false, "params"_a=std::vector<double>{}, "sigma2"_a=NaN, "log_likelihood"_a=NaN);
    size_t rows = values.size() - static_cast<size_t>(p);
    size_t cols = static_cast<size_t>(p + 1);
    std::vector<std::vector<double>> xtx(cols, std::vector<double>(cols, 0.0));
    std::vector<double> xty(cols, 0.0);
    for (size_t r = 0; r < rows; ++r) {
        size_t t = r + static_cast<size_t>(p);
        std::vector<double> x(cols, 1.0);
        for (int lag = 1; lag <= p; ++lag) x[static_cast<size_t>(lag)] = values[t - static_cast<size_t>(lag)];
        for (size_t i = 0; i < cols; ++i) {
            xty[i] += x[i] * values[t];
            for (size_t j = 0; j < cols; ++j) xtx[i][j] += x[i] * x[j];
        }
    }
    auto beta = solve_linear(xtx, xty);
    if (beta.empty() || !is_finite(beta[0])) return py::dict("success"_a=false, "params"_a=beta, "sigma2"_a=NaN, "log_likelihood"_a=NaN);
    std::vector<double> resid;
    for (size_t r = 0; r < rows; ++r) {
        size_t t = r + static_cast<size_t>(p);
        double fitted = beta[0];
        for (int lag = 1; lag <= p; ++lag) fitted += beta[static_cast<size_t>(lag)] * values[t - static_cast<size_t>(lag)];
        resid.push_back(values[t] - fitted);
    }
    double sigma2 = std::max(variance_population(resid), EPS);
    double ll = -0.5 * static_cast<double>(resid.size()) * (std::log(2.0 * PI * sigma2) + 1.0);
    int k = p + 2;
    return py::dict("success"_a=true, "params"_a=beta, "sigma2"_a=sigma2, "log_likelihood"_a=ll,
                    "aic"_a=(-2.0 * ll + 2.0 * k), "bic"_a=(-2.0 * ll + std::log(std::max<size_t>(resid.size(), 2)) * k));
}

double arma_css_loglik(const std::vector<double>& values, const std::vector<double>& ar, const std::vector<double>& ma, double intercept, std::vector<double>* out_resid=nullptr) {
    std::vector<double> resid(values.size(), 0.0);
    for (size_t t = 0; t < values.size(); ++t) {
        double fitted = intercept;
        for (size_t lag = 1; lag <= ar.size(); ++lag) if (t >= lag) fitted += ar[lag - 1] * values[t - lag];
        for (size_t lag = 1; lag <= ma.size(); ++lag) if (t >= lag) fitted += ma[lag - 1] * resid[t - lag];
        resid[t] = is_finite(values[t]) ? values[t] - fitted : NaN;
    }
    auto clean = finite_vec(resid);
    if (out_resid) *out_resid = resid;
    if (clean.empty()) return NaN;
    double sigma2 = std::max(variance_population(clean), EPS);
    return -0.5 * static_cast<double>(clean.size()) * (std::log(2.0 * PI * sigma2) + 1.0);
}

double garch_likelihood_core(const std::vector<double>& raw, double omega, double alpha, double beta, bool student, double nu) {
    auto values = finite_vec(raw);
    if (values.empty()) return NaN;
    double variance = std::max(variance_population(values), EPS);
    double ll = 0.0;
    nu = std::max(nu, 2.1);
    for (double value : values) {
        variance = std::max(omega + alpha * value * value + beta * variance, EPS);
        if (!student) {
            ll += -0.5 * (std::log(2.0 * PI * variance) + value * value / variance);
        } else {
            double z2 = value * value / variance;
            ll += std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log((nu - 2.0) * PI * variance)
                  - ((nu + 1.0) / 2.0) * std::log1p(z2 / (nu - 2.0));
        }
    }
    return ll;
}

py::dict ljung_box_statistic(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int lags) {
    auto values = finite_vec(to_vec(arr));
    int n = static_cast<int>(values.size());
    if (n <= lags + 1) return py::dict("statistic"_a=NaN, "p_value"_a=NaN, "sample_size"_a=n);
    auto rho = acf(py::array_t<double>(values.size(), values.data()), lags);
    double q = 0.0;
    for (int k = 1; k <= lags; ++k) q += (rho[static_cast<size_t>(k)] * rho[static_cast<size_t>(k)]) / static_cast<double>(n - k);
    q *= static_cast<double>(n) * static_cast<double>(n + 2);
    return py::dict("statistic"_a=q, "p_value"_a=chi2_sf_approx(q, lags), "sample_size"_a=n);
}

py::dict arch_lm_statistic(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int lags) {
    auto values = finite_vec(to_vec(arr));
    std::vector<double> sq(values.size());
    for (size_t i = 0; i < values.size(); ++i) sq[i] = values[i] * values[i];
    if (sq.size() <= static_cast<size_t>(lags + 2)) return py::dict("statistic"_a=NaN, "p_value"_a=NaN, "sample_size"_a=static_cast<int>(sq.size()));
    int rows = static_cast<int>(sq.size()) - lags;
    int cols = lags + 1;
    std::vector<std::vector<double>> xtx(static_cast<size_t>(cols), std::vector<double>(static_cast<size_t>(cols), 0.0));
    std::vector<double> xty(static_cast<size_t>(cols), 0.0);
    std::vector<double> y(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        int t = r + lags;
        y[static_cast<size_t>(r)] = sq[static_cast<size_t>(t)];
        std::vector<double> x(static_cast<size_t>(cols), 1.0);
        for (int lag = 1; lag <= lags; ++lag) x[static_cast<size_t>(lag)] = sq[static_cast<size_t>(t - lag)];
        for (int i = 0; i < cols; ++i) {
            xty[static_cast<size_t>(i)] += x[static_cast<size_t>(i)] * y[static_cast<size_t>(r)];
            for (int j = 0; j < cols; ++j) xtx[static_cast<size_t>(i)][static_cast<size_t>(j)] += x[static_cast<size_t>(i)] * x[static_cast<size_t>(j)];
        }
    }
    auto beta = solve_linear(xtx, xty);
    double ymean = mean(y), ssr = 0.0, sst = 0.0;
    for (int r = 0; r < rows; ++r) {
        int t = r + lags;
        double fitted = beta[0];
        for (int lag = 1; lag <= lags; ++lag) fitted += beta[static_cast<size_t>(lag)] * sq[static_cast<size_t>(t - lag)];
        ssr += (y[static_cast<size_t>(r)] - fitted) * (y[static_cast<size_t>(r)] - fitted);
        sst += (y[static_cast<size_t>(r)] - ymean) * (y[static_cast<size_t>(r)] - ymean);
    }
    double r2 = sst <= 0.0 ? 0.0 : std::max(0.0, 1.0 - ssr / sst);
    double stat = static_cast<double>(rows) * r2;
    return py::dict("statistic"_a=stat, "p_value"_a=chi2_sf_approx(stat, lags), "sample_size"_a=rows);
}

py::dict jarque_bera_statistic(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto values = finite_vec(to_vec(arr));
    int n = static_cast<int>(values.size());
    if (n < 3) return py::dict("statistic"_a=NaN, "p_value"_a=NaN, "sample_size"_a=n, "skewness"_a=NaN, "kurtosis"_a=NaN);
    double m = mean(values);
    double v = variance_population(values);
    double s = std::sqrt(std::max(v, 0.0));
    if (s <= 0.0) return py::dict("statistic"_a=0.0, "p_value"_a=1.0, "sample_size"_a=n, "skewness"_a=0.0, "kurtosis"_a=3.0);
    double skew = 0.0, kurt = 0.0;
    for (double value : values) {
        double z = (value - m) / s;
        skew += z * z * z;
        kurt += z * z * z * z;
    }
    skew /= static_cast<double>(n);
    kurt /= static_cast<double>(n);
    double stat = static_cast<double>(n) / 6.0 * (skew * skew + ((kurt - 3.0) * (kurt - 3.0)) / 4.0);
    return py::dict("statistic"_a=stat, "p_value"_a=chi2_sf_approx(stat, 2), "sample_size"_a=n, "skewness"_a=skew, "kurtosis"_a=kurt);
}

py::dict nelder_mead(py::function func, std::vector<double> x0, double step, int max_iter, double tol) {
    auto started = std::chrono::steady_clock::now();
    size_t n = x0.size();
    std::vector<std::pair<double, std::vector<double>>> simplex;
    simplex.push_back({py::cast<double>(func(x0)), x0});
    for (size_t i = 0; i < n; ++i) {
        auto x = x0;
        x[i] += step;
        simplex.push_back({py::cast<double>(func(x)), x});
    }
    bool converged = false;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        std::sort(simplex.begin(), simplex.end(), [](auto& a, auto& b) { return a.first < b.first; });
        double spread = std::abs(simplex.back().first - simplex.front().first);
        if (spread < tol) {
            converged = true;
            break;
        }
        std::vector<double> centroid(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) centroid[j] += simplex[i].second[j];
        }
        for (double& value : centroid) value /= static_cast<double>(n);
        auto worst = simplex.back().second;
        std::vector<double> reflected(n);
        for (size_t j = 0; j < n; ++j) reflected[j] = centroid[j] + (centroid[j] - worst[j]);
        double fr = py::cast<double>(func(reflected));
        if (fr < simplex.front().first) {
            std::vector<double> expanded(n);
            for (size_t j = 0; j < n; ++j) expanded[j] = centroid[j] + 2.0 * (reflected[j] - centroid[j]);
            double fe = py::cast<double>(func(expanded));
            simplex.back() = fe < fr ? std::make_pair(fe, expanded) : std::make_pair(fr, reflected);
        } else if (fr < simplex[n - 1].first) {
            simplex.back() = {fr, reflected};
        } else {
            std::vector<double> contracted(n);
            for (size_t j = 0; j < n; ++j) contracted[j] = centroid[j] + 0.5 * (worst[j] - centroid[j]);
            double fc = py::cast<double>(func(contracted));
            if (fc < simplex.back().first) {
                simplex.back() = {fc, contracted};
            } else {
                for (size_t i = 1; i < simplex.size(); ++i) {
                    for (size_t j = 0; j < n; ++j) simplex[i].second[j] = simplex.front().second[j] + 0.5 * (simplex[i].second[j] - simplex.front().second[j]);
                    simplex[i].first = py::cast<double>(func(simplex[i].second));
                }
            }
        }
    }
    std::sort(simplex.begin(), simplex.end(), [](auto& a, auto& b) { return a.first < b.first; });
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    return py::dict("x"_a=simplex.front().second, "fun"_a=simplex.front().first, "success"_a=converged, "max_iter"_a=max_iter,
                    "iterations"_a=iter, "fit_time_ms"_a=static_cast<double>(elapsed), "warning"_a=(converged ? "" : "max_iter reached"));
}

double rank_stability(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto values = finite_vec(to_vec(arr));
    if (values.size() <= 1) return 1.0;
    double total = 0.0;
    for (size_t i = 1; i < values.size(); ++i) total += std::abs(values[i] - values[i - 1]);
    return 1.0 / (1.0 + total / static_cast<double>(values.size() - 1));
}

py::list leaderboard_aggregate(py::list rows) {
    std::vector<py::dict> items;
    for (py::handle row : rows) items.push_back(py::cast<py::dict>(row));
    std::sort(items.begin(), items.end(), [](const py::dict& a, const py::dict& b) {
        double aq = std::numeric_limits<double>::infinity();
        double bq = std::numeric_limits<double>::infinity();
        try {
            if (a.contains("avg_qlike") && !a["avg_qlike"].is_none()) aq = py::cast<double>(a["avg_qlike"]);
        } catch (const py::cast_error&) {
            aq = std::numeric_limits<double>::infinity();
        }
        try {
            if (b.contains("avg_qlike") && !b["avg_qlike"].is_none()) bq = py::cast<double>(b["avg_qlike"]);
        } catch (const py::cast_error&) {
            bq = std::numeric_limits<double>::infinity();
        }
        if (!is_finite(aq)) aq = std::numeric_limits<double>::infinity();
        if (!is_finite(bq)) bq = std::numeric_limits<double>::infinity();
        return aq < bq;
    });
    py::list out;
    int rank = 1;
    for (auto& item : items) {
        item["rank"] = rank++;
        out.append(item);
    }
    return out;
}

std::vector<std::vector<double>> model_rank_heatmap_matrix(std::vector<std::vector<double>> ranks) {
    return ranks;
}

PYBIND11_MODULE(cmva_cpp, m) {
    m.doc() = "C++ numerical kernels for CMVA";
    m.attr("name") = "cpp";
    m.def("log_price", [](py::array_t<double, py::array::c_style | py::array::forcecast> x) { return to_array(log_price(x)); });
    m.def("log_returns", [](py::array_t<double, py::array::c_style | py::array::forcecast> x) { return to_array(log_returns(x)); });
    m.def("compute_log_returns", [](py::array_t<double, py::array::c_style | py::array::forcecast> x) { return to_array(log_returns(x)); });
    m.def("difference", [](py::array_t<double, py::array::c_style | py::array::forcecast> x) { return to_array(difference(x)); });
    m.def("first_difference", [](py::array_t<double, py::array::c_style | py::array::forcecast> x) { return to_array(difference(x)); });
    m.def("seasonal_difference", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int period) { return to_array(seasonal_difference(x, period)); }, "x"_a, "period"_a);
    m.def("rolling_mean", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "mean")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_variance", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "variance")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_std", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "std")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_standard_deviation", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "std")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_skewness", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "skew")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_kurtosis", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "kurt")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("ewma_variance", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int span) { return to_array(ewma_variance(x, span)); }, "x"_a, "span"_a);
    m.def("realized_volatility", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_apply(to_vec(x), window, min_periods, "std")); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("range_based_volatility", [](py::array_t<double, py::array::c_style | py::array::forcecast> h, py::array_t<double, py::array::c_style | py::array::forcecast> l) { return to_array(range_based_volatility(h, l)); });
    m.def("rolling_covariance", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> b, int window, int min_periods) { return to_array(rolling_pair(a, b, window, min_periods, false)); }, "left"_a, "right"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_correlation", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> b, int window, int min_periods) { return to_array(rolling_pair(a, b, window, min_periods, true)); }, "left"_a, "right"_a, "window"_a, "min_periods"_a=-1);
    m.def("acf", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int max_lag) { return to_array(acf(x, max_lag)); });
    m.def("pacf_yule_walker", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int max_lag) { return to_array(pacf_yule_walker(x, max_lag)); });
    m.def("rolling_ols_slope", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_ols(x, window, min_periods, false)); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("rolling_ols_tstat", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int window, int min_periods) { return to_array(rolling_ols(x, window, min_periods, true)); }, "x"_a, "window"_a, "min_periods"_a=-1);
    m.def("ar_fit", &ar_fit);
    m.def("ar_conditional_loglikelihood", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, py::array_t<double, py::array::c_style | py::array::forcecast> params) {
        auto values = finite_vec(to_vec(x)); auto coef = to_vec(params); int p = static_cast<int>(coef.size()) - 1;
        if (p < 0 || values.size() <= static_cast<size_t>(p + 1)) return NaN;
        std::vector<double> resid; for (size_t t = static_cast<size_t>(p); t < values.size(); ++t) { double fitted = coef[0]; for (int lag=1; lag<=p; ++lag) fitted += coef[static_cast<size_t>(lag)] * values[t-static_cast<size_t>(lag)]; resid.push_back(values[t]-fitted); }
        double sigma2 = std::max(variance_population(resid), EPS); return -0.5 * static_cast<double>(resid.size()) * (std::log(2.0 * PI * sigma2) + 1.0);
    });
    m.def("arma_css_residuals", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, py::object ar_params, py::object ma_params, double intercept) {
        std::vector<double> resid; arma_css_loglik(to_vec(x), object_vec(ar_params), object_vec(ma_params), intercept, &resid); return to_array(resid);
    }, "x"_a, "ar_params"_a=py::none(), "ma_params"_a=py::none(), "intercept"_a=0.0);
    m.def("arima_css_likelihood", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int d, py::object ar_params, py::object ma_params, double intercept) {
        auto values = to_vec(x); for (int i=0; i<std::max(0,d); ++i) { std::vector<double> diff; for (size_t j=1; j<values.size(); ++j) diff.push_back(values[j]-values[j-1]); values = diff; } return arma_css_loglik(values, object_vec(ar_params), object_vec(ma_params), intercept);
    }, "x"_a, "d"_a, "ar_params"_a=py::none(), "ma_params"_a=py::none(), "intercept"_a=0.0);
    m.def("arch_likelihood", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, double omega, double alpha) { return garch_likelihood_core(to_vec(x), omega, alpha, 0.0, false, 8.0); });
    m.def("garch_likelihood", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, double omega, double alpha, double beta) { return garch_likelihood_core(to_vec(x), omega, alpha, beta, false, 8.0); });
    m.def("student_t_garch_likelihood", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, double omega, double alpha, double beta, double nu) { return garch_likelihood_core(to_vec(x), omega, alpha, beta, true, nu); });
    m.def("qlike", [](py::array_t<double, py::array::c_style | py::array::forcecast> r, py::array_t<double, py::array::c_style | py::array::forcecast> f) { return to_array(qlike(r, f)); });
    m.def("rmse", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> f) { return loss_metric(a, f, "rmse"); });
    m.def("mae", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> f) { return loss_metric(a, f, "mae"); });
    m.def("forecast_bias", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> f) { return loss_metric(a, f, "bias"); });
    m.def("forecast_realized_correlation", &corr_metric);
    m.def("ljung_box_statistic", &ljung_box_statistic);
    m.def("arch_lm_statistic", &arch_lm_statistic);
    m.def("jarque_bera_statistic", &jarque_bera_statistic);
    m.def("rank_stability", &rank_stability);
    m.def("leaderboard_aggregate", &leaderboard_aggregate);
    m.def("model_rank_heatmap_matrix", &model_rank_heatmap_matrix);
    m.def("nelder_mead", &nelder_mead, "func"_a, "x0"_a, "step"_a=0.1, "max_iter"_a=200, "tol"_a=1e-8);
}
