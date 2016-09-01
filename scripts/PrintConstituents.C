// a ROOT macro to print out all volumes contained in a given volume ("topname")
// (used mainly for subsequent systematic stress testing of all parts of a detector -- from
//  simple to complex)

typedef std::pair<std::string, size_t> Volume_Count_t;

void FillParts(TGeoVolume const *v, std::vector<Volume_Count_t> &allparts)
{
  allparts.push_back(Volume_Count_t(
      std::string(std::string(v->GetName()) + std::string(" ") + std::string(v->GetShape()->ClassName())),
      v->GetNtotal()));

  for (size_t d = 0; d < (size_t)v->GetNdaughters(); ++d) {
    FillParts(v->GetNode(d)->GetVolume(), allparts);
  }
}

void PrintParts(std::vector<Volume_Count_t> const &allparts)
{
  size_t counter(0);
  for (auto v : allparts) {
    std::cerr << counter++ << " " << v.first << " " << v.second << "\n";
  }
}

void PrintConstituents(char const *detector = "alice.root", char const *topname = "ITSS")
{

  TGeoManager::Import(detector);
  TGeoVolume *topvolume = gGeoManager->FindVolumeFast(topname);
  if (topvolume == nullptr) {
    std::cerr << "Warning: NO SUCH VOLUME FOUND .. will use top volume\n";
    topvolume = gGeoManager->GetTopVolume();
  }

  if (topvolume == nullptr) {
    std::cerr << "ERROR: NO VOLUME FOUND .. aborting\n";
    return;
  }

  std::vector<Volume_Count_t> allparts;

  // fill parts vector
  FillParts(topvolume, allparts);

  // remove duplicates
  std::set<Volume_Count_t> allpartsset(allparts.begin(), allparts.end());

  // sort parts according to number of total nodes
  // from elementary to complex
  std::vector<Volume_Count_t> allpartsfinal;
  std::copy(allpartsset.begin(), allpartsset.end(), std::back_inserter(allpartsfinal));
  std::sort(allpartsfinal.begin(), allpartsfinal.end(),
            [](Volume_Count_t p1, Volume_Count_t p2) { return p1.second < p2.second; });

  // print parts
  PrintParts(allpartsfinal);
}
