function run_multishot_frame(input_mat, output_mat, legacy_root, archive_root)
% Run the original multi-shot PhaseMap scripts on ONE explicitly indexed frame.
% Input is [PE,RO,1,coil] complex trajectory-regridded k-space, plus measured
% LPE/LRO/a/ShiftPE/NumShots and slice/volume/echo provenance. No raw data,
% physical parameters or manual masks are synthesized by this bridge.
%
% This bridge runs the phase stages, with bAMPCorrect=0 as in the source.
% The ad-hoc unconditional amplitude block added to the old top-level wrapper
% is not a phase correction and is deliberately not part of this adapter.
addpath(genpath(legacy_root));
if nargin < 4
    project_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    archive_root = fullfile(project_dir,'..','data','spen_acquired_260915', ...
        'raw_spectroscopy','20240229_190150_cts_240229_multi_delay_spec_1_1','SPENReco');
end
if isfolder(archive_root)
    addpath(genpath(archive_root),'-end');
end
set(groot, 'defaultFigureVisible', 'off');
maxNumCompThreads(1);
settings_object = parallel.Settings;
previous_autocreate = settings_object.Pool.AutoCreate;
settings_object.Pool.AutoCreate = false;
restore_settings = onCleanup(@() restore_parallel_setting(previous_autocreate));
payload = load(input_mat);
CmplxData = double(payload.CmplxData);
assert(ndims(CmplxData) <= 4 && size(CmplxData,3) == 1);
NumShots = double(payload.NumShots);
LPE = double(payload.LPE); LRO = double(payload.LRO);
a_rad2cmsqr = double(payload.a_rad2cmsqr);
ShiftPE = double(payload.ShiftPE);
assert(NumShots > 1 && mod(size(CmplxData,1),2*NumShots) == 0);
assert(all(isfinite(CmplxData(:))));
DebugShow = false;
ProcessWithPrePhaseCorr = 1;
SmoothMotionPhaseBetweenShots = ~strcmp(payload.regrid_flavor, 'pv5');
bAMPCorrect = 0;
bUseMask = false;
bSelectedPhaseCorrRegion = false;
AssumNoiseLength = 1/14;
Std2NoiseThreshFactor = 4;
aSign = -1;
GaussRelativeWidth = .8;
ky1RelativePos = 0;
NumImages = size(CmplxData,4);
SliceNum = NumImages; ArrayNum = 1;
SliceNum2 = 1; ArrayNum2 = NumImages;
matrix_size = [size(CmplxData,2), size(CmplxData,1), 1];
xShowRegion = linspace(-LPE/2,LPE/2,size(CmplxData,1));
yShowRegion = linspace(-LRO/2,LRO/2,size(CmplxData,2));
Para.nseg = NumShots;
ParamsIn = struct('ShiftPE',ShiftPE,'ShowResults',false, ...
    'GaussRelativeWidth',GaussRelativeWidth,'ky1RelativePos',0, ...
    'a_rad2cmsqr',a_rad2cmsqr,'LPE',LPE,'aSign',aSign);
ParamsIn.RefflessParams.Polyfit2DOrder = 2;
ParamsIn.RefflessParams.Polyfit2DCoeffientsFit = ones(1,6);
ParamsIn.RefflessParams.FixSignalDirect1D = true;
roffted_original = FFTKSpace2XSpace(CmplxData,2);
if mod(NumShots,2) == 1
    if payload.echo_index > 0
        assert(isfield(payload,'first_echo_output') && isfile(payload.first_echo_output), ...
            'Later odd-shot echoes require the same slice/volume first-echo mask');
        first_echo = load(payload.first_echo_output,'MaskForSliceEcho1');
        assert(isfield(first_echo,'MaskForSliceEcho1') && any(first_echo.MaskForSliceEcho1(:)), ...
            'The first-echo automatic mask is absent or empty');
        MaskForSliceEcho1 = first_echo.MaskForSliceEcho1;
        FixAndReconRefflessMultishotHybridSPEN_Robust_NewWholeWithMask;
    else
        FixAndReconRefflessMultishotHybridSPEN_Robust_NewWhole_back;
    end
    roffted_after_prephase = FFTKSpace2XSpace(CmplxData,2);
    if payload.echo_index > 0
        FixAndReconRefflessMultishotHybridSPEN_Robust_OddNumWithMask;
    else
        FixAndReconRefflessMultishotHybridSPEN_Robust_OddNum;
    end
    roffted_corrected = WholeFixedSignalPostROFFT;
    phase_method = 'Original MATLAB NewWhole + OddNum; later echoes use same-slice/volume automatic first-echo mask';
else
    assert(mod(log2(NumShots),1) == 0, 'Original even-shot implementation requires a power of two');
    % The NormalmultiSPEN entry point uses hierarchical two-way shot merging.
    % Preserve its first-order-in-PE phase-fit restriction and defaults.
    ParamsIn.GaussRelativeWidth = .5;
    ParamsIn.RefflessParams.Polyfit2DCoeffientsFit = [1 1 0 1 0 0];
    ParamsIn = {ParamsIn, ParamsIn, ParamsIn};
    rng(20260915,'twister'); % Reproducible legacy random-pixel inverse default.
    FixAndReconRefflessMultishotHybridSPEN;
    roffted_corrected = FFTKSpace2XSpace(CmplxDataFullLatest,2);
    roffted_after_prephase = roffted_original;
    phase_method = 'Original MATLAB hierarchical power-of-two multi-shot phase correction';
end
[inva_weighted_adjoint,encoding] = calcInvA(a_rad2cmsqr,LPE, ...
    size(roffted_corrected,1),ShiftPE,1,0,GaussRelativeWidth);
inva_corrected = MultMatTensor(inva_weighted_adjoint,roffted_corrected);
assert(all(isfinite(roffted_corrected(:))) && all(isfinite(inva_corrected(:))));
effective_gauss_relative_width = GaussRelativeWidth;
mask_creation_error = '';
if payload.echo_index == 0 && mod(NumShots,2) == 1
    % Exactly the automatic mask sequence at the end of the old PV360 entry:
    % adaptive coil combine, mean smoothing, 15% threshold, largest component.
    try
        coil_images = reshape(inva_corrected,size(inva_corrected,1),size(inva_corrected,2),1,NumImages);
        first_image = abs(coilCombinebao(coil_images));
        first_image = smooth2a(first_image,5,5);
        MaskForSliceEcho1 = first_image > .15*max(first_image(:));
        MaskForSliceEcho1 = imopen(MaskForSliceEcho1,strel('disk',3));
        components = bwconncomp(MaskForSliceEcho1,4);
        assert(components.NumObjects > 0,'First-echo automatic mask has no connected component');
        sizes = cellfun(@numel,components.PixelIdxList);
        [~,largest] = max(sizes);
        MaskForSliceEcho1(:) = false;
        MaskForSliceEcho1(components.PixelIdxList{largest}) = true;
        MaskForSliceEcho1 = imclose(MaskForSliceEcho1,strel('disk',5));
        MaskForSliceEcho1 = imdilate(MaskForSliceEcho1,strel('disk',2));
    catch mask_error
        MaskForSliceEcho1 = false(size(inva_corrected,1),size(inva_corrected,2));
        mask_creation_error = mask_error.message;
    end
elseif ~exist('MaskForSliceEcho1','var')
    MaskForSliceEcho1 = [];
end
source_scan = payload.source_scan;
slice_index = payload.slice_index; volume_index = payload.volume_index;
echo_index = payload.echo_index;
amplitude_correction = 'disabled; phase-only bridge';
manual_mask_used = false;
save(output_mat,'roffted_original','roffted_after_prephase','roffted_corrected', ...
    'inva_corrected','inva_weighted_adjoint','encoding', ...
    'source_scan','slice_index','volume_index','echo_index', ...
    'phase_method','amplitude_correction','manual_mask_used', ...
    'MaskForSliceEcho1','mask_creation_error','effective_gauss_relative_width','-v7');
close all force;
end

function restore_parallel_setting(previous)
settings_object = parallel.Settings;
settings_object.Pool.AutoCreate = previous;
end
