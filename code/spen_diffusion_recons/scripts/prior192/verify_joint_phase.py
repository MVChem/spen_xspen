"""Recompute saved pilot metrics, physics, masks and network outputs on CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1].parent / 'spenpy'))
from joint_phase_diffpir import PhaseOperator
from sr_operator import SpenSuperResolutionOperator
from pilot_testtime_even_odd import TinyPhase
from phase_inva import coords_grid
from evaluate import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sim = args.run_root / 'gpu1_simulation'
    summary = json.loads((sim / 'summary.json').read_text())
    source = np.load(Path(summary['config']['source']) / 'simulation.npz', allow_pickle=False)
    assert len(summary['cases']) == 16
    errors = dict(phase_network=0., measurement_residual=0., psnr=0., ssim=0.,
                  oracle_undo=0., reference_image=0., saved_clean_forward=0.)
    method_count, cg_count = 0, 0
    for row in summary['cases']:
        data = np.load(sim / (row['name'] + '.npz'), allow_pickle=False)
        assert all(np.isfinite(data[k]).all() for k in data.files)
        mask = torch.from_numpy(data['mask'])
        assert int(mask.sum()) == (96 if row['sampling'] == 'full' else 48)
        assert not bool(torch.from_numpy(data['phase_weights'])[~mask[1::2]].any())
        base = SpenSuperResolutionOperator(torch.from_numpy(data['encoding']),
                                           torch.from_numpy(data['coils']), 96, mask)
        truth = torch.from_numpy(data['target'])[None, None] * 2 - 1
        clean_error = float((base.forward(truth)[0] - torch.from_numpy(data['clean_observation'])).abs().max())
        errors['saved_clean_forward'] = max(errors['saved_clean_forward'], clean_error)
        assert clean_error < 1e-5
        observed = torch.from_numpy(data['observation'])[None]
        net = TinyPhase()
        ckpt = torch.load(sim / (row['name'] + '_phase_net.pt'), map_location='cpu', weights_only=True)
        assert ckpt['parameter_count'] == 1153
        net.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():
            phase = net(coords_grid(48, 96, torch.device('cpu')), (48, 96))
        errors['phase_network'] = max(errors['phase_network'], float((phase - torch.from_numpy(data['phase_joint'])).abs().max()))
        assert errors['phase_network'] < 1e-4
        for method, result in row['methods'].items():
            method_count += 1
            x = torch.from_numpy(data['image_' + method])[None, None] * 2 - 1
            recomputed = metrics(x, truth)[0]
            for key in ('psnr', 'ssim'):
                errors[key] = max(errors[key], abs(recomputed[key] - result[key]))
                assert errors[key] < 1e-4
            op = PhaseOperator(base, torch.from_numpy(data['phase_' + method]))
            residual = float(op.relative_residual(x, observed))
            errors['measurement_residual'] = max(errors['measurement_residual'], abs(residual - result['measurement_nrmse']))
            assert errors['measurement_residual'] < 1e-5
            assert result['cg_nonconverged'] == 0 and result['cg_calls'] == 60
            cg_count += result['cg_calls']
            if 'max_difference_to_reference_image' in result:
                errors['reference_image'] = max(errors['reference_image'], result['max_difference_to_reference_image'])
        ci = 0 if row['sampling'] == 'full' else 1
        index = list(source['case_keys']).index(row['case_key'])
        corrected = PhaseOperator(base, torch.from_numpy(data['phase_true'])).factor.conj() * observed
        original = torch.from_numpy(source['observation'][ci, index])[None, :, mask]
        errors['oracle_undo'] = max(errors['oracle_undo'], float((corrected - original).abs().max()))
        assert errors['oracle_undo'] < 1e-6
    real_count = 0
    for folder in ('gpu1_real', 'gpu1_real_residual'):
        path = args.run_root / folder
        rows = json.loads((path / 'summary.json').read_text())['cases']
        assert len(rows) == 2
        for row in rows:
            real_count += 1
            data = np.load(path / (row['name'] + '.npz'), allow_pickle=False)
            assert all(np.isfinite(data[k]).all() for k in data.files)
            for result in row['methods'].values():
                assert 'psnr' not in result and 'ssim' not in result
                assert result['cg_nonconverged'] == 0 and result['cg_calls'] == 60
                cg_count += result['cg_calls']
                method_count += 1
    report = dict(passed=True, simulation_cases=16, real_case_conditions=real_count,
                  reconstructions=method_count, converged_cg_calls=cg_count,
                  maximum_errors=errors, torch_version=torch.__version__, numpy_version=np.__version__)
    (args.run_root / 'artifact_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
