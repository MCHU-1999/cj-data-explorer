// For the exterior orientaion:
inline bool LoadMicmacOriXml(
    const char* pszOriXml, const char * pszMicmacDir, CImgCameraIOPOri * pCameraIOP, 
    Eigen::Vector3d globalOffset, Eigen::Matrix3d & ExtrinsicRotate, Eigen::Vector3d & ExtrinsicXyz)
{
  XmlDoc doc(pszOriXml);
        
  TiXmlNode* root = doc.root("ExportAPERO");
  if(! root){
        printf("Can not open xml %s\n", pszOriXml);
        return false;
  }
        
  TiXmlNode* orientationconique = FindNode(root,"OrientationConique");
  std::string FileInterneStr = ReadNodeAsString(orientationconique, "FileInterne");
  
  char szCameraIop[512] = { 0 }; 
  sprintf(szCameraIop, "%s/%s", pszMicmacDir, FileInterneStr.c_str());

  //std::cout << "IOP: " << szCameraIop << std::endl;

  LoadMicmacCameraXml(szCameraIop, pCameraIOP);

  //std::cout << "bug" << std::endl;  inline bool LoadMicmacOriXml(
  const char* pszOriXml, const char * pszMicmacDir, CImgCameraIOPOri * pCameraIOP, 
  Eigen::Vector3d globalOffset, Eigen::Matrix3d & ExtrinsicRotate, Eigen::Vector3d & ExtrinsicXyz)
  {
    XmlDoc doc(pszOriXml);
          
    TiXmlNode* root = doc.root("ExportAPERO");
    if(! root){
          printf("Can not open xml %s\n", pszOriXml);
          return false;
    }
          
    TiXmlNode* orientationconique = FindNode(root,"OrientationConique");
    std::string FileInterneStr = ReadNodeAsString(orientationconique, "FileInterne");
    
    char szCameraIop[512] = { 0 }; 
    sprintf(szCameraIop, "%s/%s", pszMicmacDir, FileInterneStr.c_str());

    //std::cout << "IOP: " << szCameraIop << std::endl;

    LoadMicmacCameraXml(szCameraIop, pCameraIOP);

    //std::cout << "bug" << std::endl;
          
    TiXmlNode* externe = FindNode(orientationconique,"Externe");
          
    std::string CentreStr = ReadNodeAsString(externe, "Centre");
    std::stringstream CentreStream(CentreStr);
    
    double xyz[3] = { 0 };
    CentreStream >> xyz[0] >> xyz[1] >> xyz[2];
    for (int i = 0; i < 3; ++i){
      ExtrinsicXyz(i) = xyz[i];
    }
    
    ExtrinsicXyz = ExtrinsicXyz - globalOffset;
    
    TiXmlNode * paramrotation = FindNode(externe,"ParamRotation");
    TiXmlNode * codagematr = FindNode(paramrotation, "CodageMatr");
    
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
    
    // for micmac the rotate matrix is nearly symmetry
    for (int i = 0; i < 3; ++i){
      for (int j = 0; j < 3; ++j){
        ExtrinsicRotate(i, j) = r[j * 3 + i];
      }
    }

    return true;
  }
        
  TiXmlNode* externe = FindNode(orientationconique,"Externe");
        
  std::string CentreStr = ReadNodeAsString(externe, "Centre");
  std::stringstream CentreStream(CentreStr);
  
  double xyz[3] = { 0 };
  CentreStream >> xyz[0] >> xyz[1] >> xyz[2];
  for (int i = 0; i < 3; ++i){
    ExtrinsicXyz(i) = xyz[i];
  }
  
  ExtrinsicXyz = ExtrinsicXyz - globalOffset;
  
  TiXmlNode * paramrotation = FindNode(externe,"ParamRotation");
  TiXmlNode * codagematr = FindNode(paramrotation, "CodageMatr");
  
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
  
  // for micmac the rotate matrix is nearly symmetry
  for (int i = 0; i < 3; ++i){
    for (int j = 0; j < 3; ++j){
      ExtrinsicRotate(i, j) = r[j * 3 + i];
    }
  }

  return true;
}

// For the projection matrix :
Eigen::Vector3d tVectorL = -m_ExtrinsicRotate[nIndexL]*m_ExtrinsicPos[nIndexL];
Eigen::Matrix3x4d proj_matrix1 = colmap::ComposeProjectionMatrix(m_ExtrinsicRotate[nIndexL], tVectorL);