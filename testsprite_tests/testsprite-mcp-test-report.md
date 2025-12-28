# TestSprite AI Testing Report (MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** audio-cnn
- **Date:** 2025-11-20
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

### Requirement: Audio Upload & Inference Workflow
- **Description:** Users can upload WAV files, select Baseline or SE-ResNet backends, and the service must gracefully handle corrupt inputs.

#### Test TC001
- **Test Name:** Upload WAV file and perform inference using Baseline ResNet-34
- **Test Code:** [TC001_Upload_WAV_file_and_perform_inference_using_Baseline_ResNet_34.py](./TC001_Upload_WAV_file_and_perform_inference_using_Baseline_ResNet_34.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/faedcea3-9540-4ef6-b080-f9e4764e931a
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Baseline ResNet accepted the WAV upload, produced class probabilities, and surfaced them to the UI with no regressions.
---

#### Test TC002
- **Test Name:** Upload WAV file and perform inference using SE-ResNet-34 model
- **Test Code:** [TC002_Upload_WAV_file_and_perform_inference_using_SE_ResNet_34_model.py](./TC002_Upload_WAV_file_and_perform_inference_using_SE_ResNet_34_model.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/96dfe99a-c697-4445-a439-aff41ad2c3b2
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Switching to the SE-ResNet backend returns predictions within the expected confidence range, confirming multi-model routing works.
---

#### Test TC006
- **Test Name:** WAV upload with invalid or corrupted audio data
- **Test Code:** [TC006_WAV_upload_with_invalid_or_corrupted_audio_data.py](./TC006_WAV_upload_with_invalid_or_corrupted_audio_data.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/694411cb-a8cf-40d5-b43f-eba2adf9ee89
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Invalid payloads are rejected with clear UI messaging, so bad inputs cannot crash inference.
---

#### Test TC011
- **Test Name:** Model selection switching in inference API
- **Test Code:** [TC011_Model_selection_switching_in_inference_API.py](./TC011_Model_selection_switching_in_inference_API.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/3ac45a09-9d91-4851-b330-048e00fdc929
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Toggling between Baseline and SE models via API parameters consistently updates predictions, proving the selection logic is robust.
---

### Requirement: Model Performance Benchmarks
- **Description:** The system must meet accuracy, convergence-speed, and latency expectations on ESC-50 across available architectures.

#### Test TC003
- **Test Name:** Model accuracy validation on ESC-50 dataset
- **Test Code:** [TC003_Model_accuracy_validation_on_ESC_50_dataset.py](./TC003_Model_accuracy_validation_on_ESC_50_dataset.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/c93af2b7-84f8-4948-ab8c-3d2ad9d44380
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Validation accuracy matched the configured threshold, indicating training artifacts remain compatible with test data.
---

#### Test TC004
- **Test Name:** Validate SE-ResNet-34 faster convergence compared to Baseline
- **Test Code:** [TC004_Validate_SE_ResNet_34_faster_convergence_compared_to_Baseline.py](./TC004_Validate_SE_ResNet_34_faster_convergence_compared_to_Baseline.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/2f6dda41-b625-49cb-a75d-2f1f300d0f65
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Learning-curve comparison confirmed SE blocks reach the target metric several epochs earlier, so optimization assumptions still hold.
---

#### Test TC005
- **Test Name:** Real-time inference latency testing on GPU
- **Test Code:** [TC005_Real_time_inference_latency_testing_on_GPU.py](./TC005_Real_time_inference_latency_testing_on_GPU.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/4ee4496c-f3a1-43a3-9173-e1736fcb17e0
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** GPU inference stayed within the SLA window, so no regressions in batching or preprocessing overhead were observed.
---

### Requirement: Frontend Visualization Experience
- **Description:** The dashboard must render predictions, spectrograms, waveforms, and feature maps accurately after each upload.

#### Test TC007
- **Test Name:** Frontend visualization components correctness
- **Test Code:** [TC007_Frontend_visualization_components_correctness.py](./TC007_Frontend_visualization_components_correctness.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/8c90efb1-4521-4639-99be-c509c06104a3
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** UI displayed the top-3 predictions, spectrogram heatmaps, and waveform plots without rendering glitches across reruns.
---

### Requirement: Training Pipeline Reliability & Configuration
- **Description:** Training jobs must log metrics, persist checkpoints, support hyperparameter overrides, and allow remote orchestration.

#### Test TC008
- **Test Name:** Training pipeline logging and checkpoint persistence
- **Test Code:** [TC008_Training_pipeline_logging_and_checkpoint_persistence.py](./TC008_Training_pipeline_logging_and_checkpoint_persistence.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/589caabc-4315-44ad-8328-4d282a09a016
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** TensorBoard/W&B logs were emitted and checkpoints landed in the mounted volume, confirming persistence paths work.
---

#### Test TC010
- **Test Name:** Config-driven hyperparameter tuning functionality
- **Test Code:** [TC010_Config_driven_hyperparameter_tuning_functionality.py](./TC010_Config_driven_hyperparameter_tuning_functionality.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/7065679a-98fd-4e59-ae27-47fe20a468de
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** OmegaConf overrides propagated through the trainer, demonstrating that config-driven sweeps still honor the schema.
---

#### Test TC012
- **Test Name:** Remote training job triggering and monitoring
- **Test Code:** [TC012_Remote_training_job_triggering_and_monitoring.py](./TC012_Remote_training_job_triggering_and_monitoring.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/a5989408-6c9f-4382-af40-3c0c904c18be
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Modal job submission plus status polling succeeded end-to-end, so remote orchestration remains healthy.
---

### Requirement: Model Export Compliance
- **Description:** Trained artifacts must export cleanly to PyTorch `.pt` and ONNX formats for downstream deployment.

#### Test TC009
- **Test Name:** Model export validation for .pt and .onnx formats
- **Test Code:** [TC009_Model_export_validation_for_.pt_and_.onnx_formats.py](./TC009_Model_export_validation_for_.pt_and_.onnx_formats.py)
- **Test Error:** n/a
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/974b2cfc-2d3b-4dd0-82bb-10aacf80be8a/a3d3538d-ce1e-4725-a225-7547a50bb146
- **Status:** ✅ Passed
- **Severity:** LOW
- **Analysis / Findings:** Serialization produced both `.pt` and `.onnx` artifacts without shape mismatches, confirming export scripts are stable.
---

## 3️⃣ Coverage & Matching Metrics

- **100.00%** of executed tests passed.

| Requirement                               | Total Tests | ✅ Passed | ❌ Failed |
|-------------------------------------------|-------------|-----------|-----------|
| Audio Upload & Inference Workflow         | 4           | 4         | 0         |
| Model Performance Benchmarks              | 3           | 3         | 0         |
| Frontend Visualization Experience         | 1           | 1         | 0         |
| Training Pipeline Reliability & Configuration | 3       | 3         | 0         |
| Model Export Compliance                   | 1           | 1         | 0         |

---

## 4️⃣ Key Gaps / Risks
- Automated scenarios only exercised mock uploads and scripted datasets; real user recordings or noisy edge cases may still expose preprocessing issues.
- GPU latency checks ran against a single hardware profile; behaviour on CPU-only or lower-tier GPUs is unverified.
- Remote Modal orchestration passed, but long-running training stability ( >3h ) and failure recovery paths were not covered in this run.

---
