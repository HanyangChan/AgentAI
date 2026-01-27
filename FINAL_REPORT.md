# 🏭 Final Project Report: Hybrid Vision-AI for Semiconductor Quality Control

## 📌 Overview
This project addresses the challenge of high-precision semiconductor quality inspection by combining the industrial efficiency of **ResNet18** with the advanced semantic reasoning of **GPT-4o-mini**.

## 🏗 Architecture
The system utilizes a hierarchical pipeline:
1.  **Vision Preprocessing**: Multi-view cropping (Full, Top-Body, Bottom-Leads).
2.  **CNN Screening**: ResNet18 provides probabilistic labels and confidence metrics.
3.  **VLM Reasoning**: An LLM acts as the "Decision Head," weighing the CNN's predictions against direct visual evidence to produce a final judgment and a human-readable justification.

## 🚀 Key Achievements
- **Adaptive Detection**: Successfully identifies defects (e.g., bent leads) even when the primary CNN model reports normal status.
- **Full Traceability**: Generates a comprehensive `agent_resnet_vlm_log.txt` detailing the reasoning path for every inspected component.
- **Robust Pipeline**: Handles image downloading, multi-stage processing, and structured output generation autonomously.

## 📊 Evaluation Data
- **Dataset**: `dev.csv` (20 samples)
- **Primary Model**: ResNet18 (Baseline)
- **Secondary Model**: GPT-4o-mini (Multimodal)
- **Final Output**: `submission_resnet_vlm.csv`

## 🔗 Repository
Source code and datasets are maintained at:
[https://github.com/HanyangChan/AgentAI](https://github.com/HanyangChan/AgentAI)

---
*Created by Antigravity AI Agent*
