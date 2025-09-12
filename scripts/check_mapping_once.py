from src.common.constants import WATERSHED_TO_BASE_CLASS, CLASSIFICATION_SCHEMES

ws_vals = [0,11,12,13,21,22,23,24,25,31,32,50,255]
base = {v: WATERSHED_TO_BASE_CLASS.get(v, None) for v in ws_vals}
scheme = CLASSIFICATION_SCHEMES["3class_org"]
to_train = {}
for v in ws_vals:
    b = WATERSHED_TO_BASE_CLASS.get(v, None)
    if b is None:
        to_train[v] = "N/A"
    else:
        to_train[v] = scheme["mapping"].get(b, scheme.get("default_for_others", 255))

print("WS->BASE:", base)
print("WS->TRAIN(3c):", to_train)
