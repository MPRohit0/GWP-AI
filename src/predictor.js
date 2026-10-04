import * as ort from "onnxruntime-web";

let sessionPromise = null;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(
      `${import.meta.env.BASE_URL}model.onnx`,
      { executionProviders: ["wasm"] }
    );
  }
  return sessionPromise;
}

function numberOrZero(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/*
 * MUST MATCH THE PYTHON TRAINING CODE:
 *
 * data = data.drop(["Building ID"], axis=1)
 * data = pd.get_dummies(
 *     data,
 *     columns=["Building Type"],
 *     drop_first=True,
 *     dtype=int
 * )
 *
 * Pandas sorts the categorical dummy columns alphabetically here, with
 * Commercial as the dropped reference category:
 *
 * Building Type_Industrial
 * Building Type_Institutional
 * Building Type_Office
 * Building Type_Residential
 */
export function formToFeatureVector(form) {
  const type = form.buildingType;

  return [
    numberOrZero(form.builtUpArea),
    numberOrZero(form.floors),
    numberOrZero(form.height),
    numberOrZero(form.serviceLife),
    numberOrZero(form.cement),
    numberOrZero(form.steel),
    numberOrZero(form.bricks),
    numberOrZero(form.sand),
    numberOrZero(form.aggregate),
    numberOrZero(form.glass),
    numberOrZero(form.wood),
    numberOrZero(form.transportDistance),
    numberOrZero(form.constructionEnergy),
    numberOrZero(form.annualElectricity),

    type === "Industrial" ? 1 : 0,
    type === "Institutional" ? 1 : 0,
    type === "Office" ? 1 : 0,
    type === "Residential" ? 1 : 0
  ];
}

export async function predictGWP(form) {
  const session = await getSession();
  const features = formToFeatureVector(form);

  const input = new ort.Tensor(
    "float32",
    Float32Array.from(features),
    [1, features.length]
  );

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const results = await session.run({
    [inputName]: input
  });

  return Number(results[outputName].data[0]);
}
