
{
  "tests": [
    {
      "name": "background_removal_test",
      "input": {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Sample_product_photo.jpg"
      },
      "timeout": 20000
    }
  ],
  "config": {
    "gpuTypeId": "NVIDIA T4",
    "gpuCount": 1,
    "allowedCudaVersions": ["12.1", "12.2", "12.3"]
  }
}
