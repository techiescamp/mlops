## Commands to deploy helm chart

### Deploy kserve-inference-service

helm install kserve-inference-service kserve-inference-service -n kserve

### Deploy backend

helm install backend backend -n predictor-backend --create-namespace

### Deploy frontend

helm install frontend frontend -n predictor-frontend --create-namespace

## Commands to delete helm chart

### Delete frontend

helm delete frontend -n predictor-frontend

### Delete backend

helm delete backend -n predictor-backend

### Delete kserve-inference-service

helm delete kserve-inference-service -n kserve
