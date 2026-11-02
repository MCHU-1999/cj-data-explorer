// Load the exterior orientation (pose) of one image from a MicMac
// orientation file (e.g. Ori-XXX/Orientation-IMG_0001.JPG.xml), and the
// interior orientation (camera calibration) that it points to.
//
// Outputs:
//   pCameraIOP      - intrinsics, read from the file referenced by <FileInterne>
//   ExtrinsicRotate - world -> camera rotation (transpose of MicMac's matrix)
//   ExtrinsicXyz    - camera centre in world coordinates, minus globalOffset
inline bool LoadMicmacOriXml(
  const char* pszOriXml, const char* pszMicmacDir, CImgCameraIOPOri* pCameraIOP,
  Eigen::Vector3d globalOffset, Eigen::Matrix3d& ExtrinsicRotate, Eigen::Vector3d& ExtrinsicXyz)
{
  XmlDoc doc(pszOriXml);

  TiXmlNode* root = doc.root("ExportAPERO");
  if (!root) {
    printf("Can not open xml %s\n", pszOriXml);
    return false;
  }

  // --- Interior orientation: follow the link to the calibration file ---
  TiXmlNode* orientationconique = FindNode(root, "OrientationConique");
  std::string FileInterneStr = ReadNodeAsString(orientationconique, "FileInterne");

  char szCameraIop[512] = { 0 };
  sprintf(szCameraIop, "%s/%s", pszMicmacDir, FileInterneStr.c_str());

  LoadMicmacCameraXml(szCameraIop, pCameraIOP);

  // --- Exterior orientation ---
  TiXmlNode* externe = FindNode(orientationconique, "Externe");

  // Camera centre (projection centre) in world coordinates
  std::string CentreStr = ReadNodeAsString(externe, "Centre");
  std::stringstream CentreStream(CentreStr);

  double xyz[3] = { 0 };
  CentreStream >> xyz[0] >> xyz[1] >> xyz[2];
  for (int i = 0; i < 3; ++i) {
    ExtrinsicXyz(i) = xyz[i];
  }

  // Shift to a local origin to keep numbers small (e.g. Lambert-93 coords)
  ExtrinsicXyz = ExtrinsicXyz - globalOffset;

  // Rotation matrix, stored row by row as L1, L2, L3
  TiXmlNode* paramrotation = FindNode(externe, "ParamRotation");
  TiXmlNode* codagematr = FindNode(paramrotation, "CodageMatr");

  double r[9] = { 0 };
  std::string L1Str = ReadNodeAsString(codagematr, "L1");
  std::stringstream L1Stream(L1Str);
  L1Stream >> r[0] >> r[1] >> r[2];

  std::string L2Str = ReadNodeAsString(codagematr, "L2");
  std::stringstream L2Stream(L2Str);
  L2Stream >> r[3] >> r[4] >> r[5];

  std::string L3Str = ReadNodeAsString(codagematr, "L3");
  std::stringstream L3Stream(L3Str);
  L3Stream >> r[6] >> r[7] >> r[8];

  // MicMac stores camera -> world. Transpose it to get world -> camera.
  // (Original comment said "nearly symmetric" -- it's really a transpose.)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ExtrinsicRotate(i, j) = r[j * 3 + i];
    }
  }

  return true;
}

// --- Usage snippet (from inside some class method, not standalone) ---
// Build the 3x4 extrinsic matrix [R | t] in COLMAP convention:
//   x_cam = R * X_world + t,  with  t = -R * C
Eigen::Vector3d tVectorL = -m_ExtrinsicRotate[nIndexL] * m_ExtrinsicPos[nIndexL];
Eigen::Matrix3x4d proj_matrix1 = colmap::ComposeProjectionMatrix(m_ExtrinsicRotate[nIndexL], tVectorL);