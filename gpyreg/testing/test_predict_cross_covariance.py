from types import MethodType

import numpy as np
import pytest

import gpyreg as gpr
import gpyreg.gaussian_process as gaussian_process
from gpyreg.gaussian_process import _can_retain_cross_covariance
from gpyreg.isotropic_covariance_functions import (
    MaternIsotropic,
    SquaredExponentialIsotropic,
)


def _make_gp(covariance, noise_std=1e-2, sample_count=2):
    """Construct a small deterministic GP without fitting hyperparameters."""
    X = np.array(
        [
            [-1.0, 0.2],
            [-0.25, -0.6],
            [0.4, 0.7],
            [1.1, -0.1],
        ]
    )
    y = np.array([[-0.7], [0.1], [0.8], [0.3]])
    gp = gpr.GP(
        D=2,
        covariance=covariance,
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    cov_N = covariance.hyperparameter_count(2)
    hyp = np.zeros((sample_count, cov_N + 1))
    for sample in range(sample_count):
        hyp[sample, :cov_N] = 0.05 * sample * np.arange(1, cov_N + 1)
        hyp[sample, -1] = np.log(noise_std)
    gp.update(X_new=X, y_new=y, hyp=hyp)
    return gp


_X_STAR = np.array([[-0.7, 0.5], [0.2, -0.3], [0.9, 0.4]])
_Y_STAR = np.array([[-0.2], [0.4], [0.5]])


@pytest.mark.parametrize("noise_std, l_chol", [(1e-2, True), (1e-4, False)])
@pytest.mark.parametrize("sample_count", [1, 3])
@pytest.mark.parametrize("return_lpd", [False, True])
@pytest.mark.parametrize("add_noise", [False, True])
@pytest.mark.parametrize("separate_samples", [False, True])
def test_predict_cross_covariance_return_combinations(
    noise_std,
    l_chol,
    sample_count,
    return_lpd,
    add_noise,
    separate_samples,
):
    gp = _make_gp(
        gpr.covariance_functions.SquaredExponential(),
        noise_std=noise_std,
        sample_count=sample_count,
    )
    assert all(bool(posterior.L_chol) == l_chol for posterior in gp.posteriors)

    kwargs = {
        "y_star": _Y_STAR if return_lpd else None,
        "return_lpd": return_lpd,
        "add_noise": add_noise,
        "separate_samples": separate_samples,
    }
    baseline = gp.predict(_X_STAR, **kwargs)
    keys_before = set(gp.__dict__)
    temporary_data_before = gp.temporary_data.copy()
    actual = gp.predict(_X_STAR, return_cross_covariance=True, **kwargs)

    assert len(actual) == len(baseline) + 1
    for baseline_value, actual_value in zip(baseline, actual[:-1]):
        assert np.array_equal(actual_value, baseline_value)

    cross_covariance = actual[-1]
    assert isinstance(cross_covariance, tuple)
    assert len(cross_covariance) == sample_count
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    for sample, matrix in enumerate(cross_covariance):
        expected = gp.covariance.compute(
            gp.posteriors[sample].hyp[:cov_N], gp.X, _X_STAR
        )
        assert matrix.shape == (gp.X.shape[0], _X_STAR.shape[0])
        assert np.array_equal(matrix, expected)

    assert set(gp.__dict__) == keys_before
    assert gp.temporary_data == temporary_data_before
    assert all(
        value is not matrix
        for value in gp.__dict__.values()
        for matrix in cross_covariance
    )


@pytest.mark.parametrize(
    "covariance",
    [
        gpr.covariance_functions.SquaredExponential(),
        gpr.covariance_functions.Matern(3),
        gpr.covariance_functions.RationalQuadraticARD(),
        MaternIsotropic(3),
        SquaredExponentialIsotropic(),
    ],
)
def test_bundled_covariances_are_zero_copy_eligible(covariance):
    assert _can_retain_cross_covariance(covariance)
    gp = _make_gp(covariance)
    *_, cross_covariance = gp.predict(
        _X_STAR, separate_samples=True, return_cross_covariance=True
    )
    cov_N = covariance.hyperparameter_count(gp.D)
    for sample, matrix in enumerate(cross_covariance):
        expected = covariance.compute(
            gp.posteriors[sample].hyp[:cov_N], gp.X, _X_STAR
        )
        assert np.array_equal(matrix, expected)


def test_zero_copy_path_returns_the_matrix_used_by_prediction(monkeypatch):
    covariance = gpr.covariance_functions.SquaredExponential()
    gp = _make_gp(covariance, sample_count=3)
    covariance_type = type(covariance)
    original = covariance_type.compute
    computed_cross_matrices = []

    def recording_compute(
        self,
        hyp,
        X,
        X_star=None,
        compute_diag=False,
        compute_grad=False,
    ):
        result = original(self, hyp, X, X_star, compute_diag, compute_grad)
        if X_star is not None:
            computed_cross_matrices.append(result)
        return result

    monkeypatch.setattr(covariance_type, "compute", recording_compute)
    monkeypatch.setitem(
        gaussian_process._ZERO_COPY_CROSS_COVARIANCE_COMPUTES,
        covariance_type,
        recording_compute,
    )
    assert _can_retain_cross_covariance(covariance)

    *_, cross_covariance = gp.predict(
        _X_STAR, separate_samples=True, return_cross_covariance=True
    )

    assert len(computed_cross_matrices) == len(gp.posteriors)
    assert all(
        returned is computed
        for returned, computed in zip(
            cross_covariance, computed_cross_matrices
        )
    )


@pytest.mark.parametrize("with_training_inputs", [False, True])
@pytest.mark.parametrize("return_lpd", [False, True])
def test_prior_only_predict_returns_none_per_sample(
    with_training_inputs, return_lpd
):
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = np.array([[0.0, 0.0, 0.1, -2.0], [0.2, -0.1, 0.0, -1.5]])
    if with_training_inputs:
        gp.update(X_new=np.array([[-1.0, 0.0], [1.0, 0.5]]), hyp=hyp)
    else:
        gp.update(hyp=hyp)

    kwargs = {
        "y_star": _Y_STAR if return_lpd else None,
        "return_lpd": return_lpd,
    }
    baseline = gp.predict(_X_STAR, **kwargs)
    result = gp.predict(
        _X_STAR,
        return_cross_covariance=True,
        **kwargs,
    )
    for baseline_value, actual_value in zip(baseline, result[:-1]):
        assert np.array_equal(actual_value, baseline_value)
    assert result[-1] == (None, None)


@pytest.mark.parametrize("clear_method", ["clean", "update"])
def test_missing_factors_remain_unsupported(clear_method):
    gp = _make_gp(gpr.covariance_functions.SquaredExponential())
    if clear_method == "clean":
        gp.clean()
    else:
        gp.update(compute_posterior=False)
    posteriors = gp.posteriors
    hypotheses = [posterior.hyp.copy() for posterior in posteriors]
    with pytest.raises(TypeError) as baseline:
        gp.predict(_X_STAR)
    with pytest.raises(type(baseline.value)) as requested:
        gp.predict(_X_STAR, return_cross_covariance=True)
    assert str(requested.value) == str(baseline.value)
    assert gp.posteriors is posteriors
    for posterior, hyp in zip(posteriors, hypotheses):
        np.testing.assert_array_equal(posterior.hyp, hyp)
        for field in ("alpha", "L", "L_chol", "sW", "sn2_mult"):
            assert getattr(posterior, field) is None


def test_return_cross_covariance_is_keyword_only():
    gp = _make_gp(gpr.covariance_functions.SquaredExponential())
    with pytest.raises(TypeError):
        gp.predict(_X_STAR, None, None, False, False, False, True)


def test_cross_covariance_with_user_provided_prediction_noise():
    X = np.array([[-1.0, 0.2], [-0.25, -0.6], [0.4, 0.7], [1.1, -0.1]])
    y = np.array([[-0.7], [0.1], [0.8], [0.3]])
    s2 = np.array([[0.02], [0.03], [0.01], [0.04]])
    covariance = gpr.covariance_functions.SquaredExponential()
    gp = gpr.GP(
        D=2,
        covariance=covariance,
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(
            user_provided_add=True, scale_user_provided=True
        ),
    )
    hyp = np.array([[0.0, 0.1, -0.2, np.log(1.5)]])
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    s2_star = np.array([[0.03], [0.01], [0.02]])

    baseline = gp.predict(
        _X_STAR,
        _Y_STAR,
        s2_star,
        add_noise=True,
        return_lpd=True,
    )
    actual = gp.predict(
        _X_STAR,
        _Y_STAR,
        s2_star,
        add_noise=True,
        return_lpd=True,
        return_cross_covariance=True,
    )

    for baseline_value, actual_value in zip(baseline, actual[:-1]):
        assert np.array_equal(actual_value, baseline_value)
    expected = covariance.compute(hyp[0, :3], gp.X, _X_STAR)
    assert np.array_equal(actual[-1][0], expected)


def test_cross_covariance_preserves_lpd_input_error():
    gp = _make_gp(gpr.covariance_functions.SquaredExponential())
    with pytest.raises(ValueError, match="without y_star"):
        gp.predict(
            _X_STAR,
            return_lpd=True,
            return_cross_covariance=True,
        )


class _ScratchKernel(gpr.covariance_functions.SquaredExponential):
    """Custom kernel that reuses its cross-kernel output buffer."""

    def __init__(self):
        self.scratch = None
        self.cross_values = []
        self.cross_calls = 0

    def compute(
        self,
        hyp,
        X,
        X_star=None,
        compute_diag=False,
        compute_grad=False,
    ):
        result = super().compute(hyp, X, X_star, compute_diag, compute_grad)
        if X_star is None:
            return result
        self.cross_calls += 1
        self.cross_values.append(np.array(result, copy=True))
        if self.scratch is None:
            self.scratch = np.empty_like(result)
        np.copyto(self.scratch, result)
        return self.scratch


def test_custom_scratch_kernel_is_snapshotted_without_recomputation():
    covariance = _ScratchKernel()
    gp = _make_gp(covariance, sample_count=3)
    assert not _can_retain_cross_covariance(covariance)

    *_, cross_covariance = gp.predict(
        _X_STAR, separate_samples=True, return_cross_covariance=True
    )

    assert covariance.cross_calls == len(gp.posteriors)
    for matrix, expected in zip(cross_covariance, covariance.cross_values):
        assert np.array_equal(matrix, expected)
        assert matrix is not covariance.scratch
    assert not np.shares_memory(cross_covariance[0], cross_covariance[1])


@pytest.mark.parametrize("override_scope", ["instance", "class"])
def test_overridden_bundled_compute_is_snapshotted(
    monkeypatch, override_scope
):
    covariance = gpr.covariance_functions.SquaredExponential()
    gp = _make_gp(covariance)
    original = type(covariance).compute
    returned_cross_matrices = []

    def overridden(
        self,
        hyp,
        X,
        X_star=None,
        compute_diag=False,
        compute_grad=False,
    ):
        result = original(self, hyp, X, X_star, compute_diag, compute_grad)
        if X_star is not None:
            returned_cross_matrices.append(result)
        return result

    if override_scope == "instance":
        covariance.compute = MethodType(overridden, covariance)
    else:
        monkeypatch.setattr(type(covariance), "compute", overridden)
    assert not _can_retain_cross_covariance(covariance)

    *_, cross_covariance = gp.predict(
        _X_STAR, separate_samples=True, return_cross_covariance=True
    )

    assert len(returned_cross_matrices) == len(gp.posteriors)
    for matrix, computed in zip(cross_covariance, returned_cross_matrices):
        assert np.array_equal(matrix, computed)
        assert matrix is not computed
