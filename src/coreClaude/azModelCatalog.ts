export interface BundledAZModel {
  id: string;
  label: string;
  asset: number;
}

// Add new ONNX files here when you want selectable models in-app.
// Example:
// { id: 'iter_0099', label: 'AZ iter_0099', asset: require('../../assets/models/iter_0099.onnx') }
const BUNDLED_MODELS: BundledAZModel[] = [
  {
    id: 'iter_0079',
    label: 'AZ iter_0079',
    asset: require('../../assets/models/iter_0079.onnx'),
  },
  {
    id: 'makhos_az',
    label: 'AZ Default',
    asset: require('../../assets/models/makhos_az.onnx'),
  },
];

export function getBundledAZModels(): BundledAZModel[] {
  return BUNDLED_MODELS;
}
