
## Live Demo
Deployed on AWS EC2 (Mumbai region) with models stored on S3.

- Interactive API docs: http://13.235.104.167:8001/docs
- Prediction endpoint:  POST http://13.235.104.167:8001/predict

### Example request
curl -X POST http://13.235.104.167:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"smiles": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"}'


🧬 Multi-Target QSAR Predictor API (EGFR & BRAF)A production-ready, containerized FastAPI microservice that predicts the inhibitory activity of compounds against two critical oncology targets: EGFR and BRAF (CHEMBL5145).

This API goes beyond standard classification by implementing an Applicability Domain (AD) filter to actively intercept and flag structurally alien molecules, preventing mathematical extrapolation errors common in tree-based machine learning models.

🔬 The Cheminformatics ChallengeIn real-world drug discovery, high-throughput screening (HTS) data is severely imbalanced. The raw BRAF extraction from the ChEMBL database yielded:Actives ($\le 1000$ nM): 4,579 compounds (94.6%)Inactives ($\ge 10000$ nM): 261 compounds (5.4%)A naive machine learning model trained on this data will simply guess "Active" for entirely novel, unrelated chemotypes (e.g., predicting that Aspirin is a highly active BRAF kinase inhibitor).

In a wet-lab setting, this false-positive extrapolation wastes weeks of valuable bench time and expensive reagents on dead-end synthesis routes.

🛠️ Architectural Solutions Engineered

1. The Applicability Domain "Bouncer"To prevent extrapolation, the API calculates the Cosine Similarity of the query molecule's 1024-bit Morgan Fingerprint against the raw training dataset.

If a query molecule (like Aspirin) falls below a similarity threshold of 0.3, the API intercepts the request and returns a "reliability": "Low confidence" warning, preventing the Random Forest from blindly guessing on out-of-domain chemistry.

2. Hybrid Featurization & ScalingInputs are dynamically processed via RDKit into a 1030-feature array, combining:1D Physicochemical Descriptors: MolWt, LogP, TPSA, HBA, HBD, Rotatable Bonds.2D Morgan Fingerprints: 1024 bits (Radius 2).
Features are normalized in real-time using a pre-fitted StandardScaler.


3. Strict Algorithmic PenaltiesTo handle the 95/5 class imbalance, the BRAF Random Forest was trained utilizing strict class_weight="balanced" parameters. 
This mathematically penalized the algorithm for misclassifying the rare Inactive class, pushing the model to a highly precise baseline that prioritizes limiting false-positive synthesis candidates.
