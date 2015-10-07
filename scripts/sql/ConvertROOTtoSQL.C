int Box_Shape(int id, TGeoBBox *box);
int Tube_Shape(int id, TGeoTubeSeg *tube);
int Tube_ShapeN(int id, TGeoTube *tube);
int Cone_Shape(int id, TGeoConeSeg *cone);
int Cone_ShapeN(int id, TGeoCone *cone);
int Torus_Shape(int id, TGeoTorus *torus);
int Arb8_Shape(int id, TGeoArb8 *arb8);
int Trap_Shape(int id, TGeoTrap *trap);
int Polycone_Shape(int id, TGeoPcon *pcon);
int Polyhedra_Shape(int id, TGeoPgon *pgon);
int Trd_Shape(int id, TGeoTrd2 *trd2);
int BooleanSolid_Shape(int id, TGeoCompositeShape *comp, int depth, int *m_id, char* geometry);

void Daughter_Volume(TGeoVolume *vol, int parent_id, int *m_id);

// main function filling an SQL database out of a ROOT geometry file
void ConvertROOTtoSQL(char* geometry=NULL, int start_num= 1)
{
  if( geometry == NULL)
  {
    printf("\nUsage: ex)\n");
    printf("  root -q -b -l 'rootMacro.c(\"cms2015\")'\n");
    printf("  root -q -b -l 'rootMacro.c(\"cms2015\", 1012)'\n\n");
    return ;
  }
  char* rootfilename= (char*)malloc(strlen(geometry)+strlen(".root")+1);
  strcpy(rootfilename, geometry);
  strcat(rootfilename, ".root");
  
  if( TGeoManager::Import(rootfilename) == NULL )
  {
    exit(0);
  }


  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  bool check_flag= 0;
  int *m_id= (int*)malloc(sizeof(int));
  *m_id = 1; //
  int id= start_num;  // ID is started from start_number

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out);
  outFile_volume.close();



  TList *mlist = gGeoManager->GetListOfMaterials();
  TGeoMixture *mix = (TGeoMixture*)mlist->First(); 
  TGeoElementTable *elementTable= gGeoManager->GetElementTable();

  // Get Elements from ElementTable
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  for(unsigned int i= 0; i< elementTable->GetNelements(); i++)
  {
    outFile_volume<< setprecision(15)
		  << "insert into Element "
		  << "(ElementName, A, N, Z, Nisotopes ) values ("
		  << "'"<< elementTable->GetElement(i)->GetName() << "', "
		  << elementTable->GetElement(i)->A() << ", "
		  << elementTable->GetElement(i)->N() << ", "
		  << elementTable->GetElement(i)->Z() << ", "
		  << elementTable->GetElement(i)->GetNisotopes() << ") "
      
		  << "on duplicate key update ElementName='"
		  << elementTable->GetElement(i)->GetName()<< "';\n";
  }
  outFile_volume.close();

  
  for(unsigned int i=0; i < mlist->GetSize(); ++i)
  {
    mix = (TGeoMixture*)mlist->At( i ); 
    outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
    

    // ** Material Table
    //+--------------+-------------+------+-----+---------+----------------+
    //| Field        | Type        | Null | Key | Default | Extra          |
    //+--------------+-------------+------+-----+---------+----------------+
    //| MaterialId   | int(11)     | NO   | PRI | NULL    | auto_increment |
    //| MaterialName | varchar(50) | YES  | UNI | NULL    |                |
    //| A            | double      | YES  |     | NULL    |                |
    //| Z            | double      | YES  |     | NULL    |                |
    //| Density      | double      | YES  |     | NULL    |                |
    //| Pressure     | double      | YES  |     | NULL    |                |
    //| Temperature  | double      | YES  |     | NULL    |                |
    //| RadLen       | double      | YES  |     | NULL    |                |
    //| IntLen       | double      | YES  |     | NULL    |                |
    //| Nelements    | int(11)     | YES  |     | NULL    |                |
    //| IsMixture    | int(11)     | YES  |     | NULL    |                |
    //+--------------+-------------+------+-----+---------+----------------+

    // Add Material information
    outFile_volume<< setprecision(15)
		  << "insert into Material (MaterialName, A, Z, Density, "
		  << "Pressure, Temperature, RadLen, IntLen, Nelements, IsMixture)"
		  << "values ("
		  << "'"<< mix->GetName() << "', "
		  << mix->GetA() << ", "
		  << mix->GetZ() << ", "
		  << mix->GetDensity() << ", "
		  << mix->GetPressure() << ", "
		  << mix->GetTemperature() << ", "
		  << mix->GetRadLen() << ", "
		  << mix->GetIntLen() << ", "
		  << mix->GetNelements() << ", "
		  << mix->IsMixture() << ");\n";


    // ** Element Table
    //+-------------+-------------+------+-----+---------+----------------+
    //| Field       | Type        | Null | Key | Default | Extra          |
    //+-------------+-------------+------+-----+---------+----------------+
    //| ElementId   | int(11)     | NO   | PRI | NULL    | auto_increment |
    //| ElementName | varchar(50) | YES  | UNI | NULL    |                |
    //| A           | double      | YES  |     | NULL    |                |
    //| N           | int(11)     | YES  |     | NULL    |                |
    //| Z           | int(11)     | YES  |     | NULL    |                |
    //| Nisotopes   | int(11)     | YES  |     | NULL    |                |
    //+-------------+-------------+------+-----+---------+----------------+

    // Add Element information For Material
    if( !mix->IsMixture() )
    {
      outFile_volume<< setprecision(15)
		    << "insert into Element "
		    << "(ElementName, A, N, Z, Nisotopes ) values ("
		    << "'"<< mix->GetElement()->GetName() << "', "
		    << mix->GetElement()->A() << ", "
		    << mix->GetElement()->N() << ", "
		    << mix->GetElement()->Z() << ", "
		    << mix->GetElement()->GetNisotopes() << ") "

		    << "on duplicate key update ElementName='"
		    << mix->GetElement()->GetName()<< "';\n";


      // ** ElementsInMaterial Table
      //+------------+---------+------+-----+---------+-------+
      //| Field      | Type    | Null | Key | Default | Extra |
      //+------------+---------+------+-----+---------+-------+
      //| MaterialId | int(11) | YES  |     | NULL    |       |
      //| ElementId  | int(11) | YES  |     | NULL    |       |
      //| Weight     | double  | YES  |     | NULL    |       |
      //+------------+---------+------+-----+---------+-------+

      // Add relation between Material and Element
      outFile_volume<< "insert into ElementsInMaterial "
		    << "(MaterialId, ElementId) select "
	
		    << "(select MaterialId from Material where MaterialName='"
		    << mix->GetName() << "'), "
	
		    << "(select ElementId from Element where ElementName='"
		    << mix->GetElement()->GetName() << "')"
	
		    << " from dual where not exists "
		    << "(select * from ElementsInMaterial where "
		    << "MaterialId=(select MaterialId from Material where MaterialName='"
		    << mix->GetName() << "') && "
	
		    << "ElementId=(select ElementId from Element where ElementName='"
		    << mix->GetElement()->GetName() << "'));\n";
    }

    // Add Element information For Mixture
    if( mix->IsMixture() )
    {
      for(unsigned int j=0; j< mix->GetNelements(); j++)
      {
	outFile_volume<< setprecision(15)
		      << "insert into Element "
		      << "(ElementName, A, N, Z, Nisotopes ) values ("
		      << "'"<< mix->GetElement(j)->GetName() << "', "
		      << mix->GetElement()->A() << ", "
		      << mix->GetElement()->N() << ", "
		      << mix->GetElement()->Z() << ", "
		      << mix->GetElement(j)->GetNisotopes() << ") "
	  
		      << "on duplicate key update ElementName='"
		      << mix->GetElement(j)->GetName()<< "';\n";


	outFile_volume<< setprecision(15)
		      << "insert into ElementsInMaterial "
		      << "(MaterialId, ElementId, Weight) select "
	  
		      << "(select MaterialId from Material where MaterialName='"
		      << mix->GetName() << "'), "
	  
		      << "(select ElementId from Element where ElementName='"
		      << mix->GetElement(j)->GetName() << "'), "
	  
		      << ( mix->GetWmixt() )[j]
	  
		      << " from dual where not exists "
		      << "(select * from ElementsInMaterial where "
		      << "MaterialId=(select MaterialId from Material where MaterialName='"
		      << mix->GetName() << "') && "
	  
		      << "ElementId=(select ElementId from Element where ElementName='"
		      << mix->GetElement(j)->GetName() << "') && "
	  
		      << "Weight="
		      << ( mix->GetWmixt() )[j] << ");\n";
	

	// ** Isotope Table
	//+-------------+-------------+------+-----+---------+----------------+
	//| Field       | Type        | Null | Key | Default | Extra          |
	//+-------------+-------------+------+-----+---------+----------------+
	//| IsotopeId   | int(11)     | NO   | PRI | NULL    | auto_increment |
	//| IsotopeName | varchar(50) | YES  | UNI | NULL    |                |
	//| A           | double      | YES  |     | NULL    |                |
	//| N           | int(11)     | YES  |     | NULL    |                |
	//| Z           | int(11)     | YES  |     | NULL    |                |
	//+-------------+-------------+------+-----+---------+----------------+

	// Add Isotope information For Mixture
	for(unsigned int k=0; k< mix->GetElement(j)->GetNisotopes(); k++)
	{
	  outFile_volume<< setprecision(15)
			<< "insert into Isotope "
			<< "(IsotopeName, A, N, Z) values ("
			<< "'"<< mix->GetElement(j)->GetIsotope(k)->GetName()<< "', "
			<< mix->GetElement(j)->GetIsotope(k)->GetA() << ", "
			<< mix->GetElement(j)->GetIsotope(k)->GetN() << ", "
			<< mix->GetElement(j)->GetIsotope(k)->GetZ() << ") "

			<< "on duplicate key update IsotopeName='"
			<< mix->GetElement(j)->GetIsotope(k)->GetName()<< "';\n";


	  // ** IsotopesInElement Table
	  //+-------------------+---------+------+-----+---------+-------+
	  //| Field             | Type    | Null | Key | Default | Extra |
	  //+-------------------+---------+------+-----+---------+-------+
	  //| ElementId         | int(11) | YES  |     | NULL    |       |
	  //| IsotopeId         | int(11) | YES  |     | NULL    |       |
	  //| RelativeAbundance | double  | YES  |     | NULL    |       |
	  //+-------------------+---------+------+-----+---------+-------+

	  // Add relation between Element and Isotope
	  outFile_volume<< setprecision(15)
			<< "insert into IsotopesInElement "
			<< "(ElementId, IsotopeId, RelativeAbundance) select "
	    
			<< "(select ElementId from Element where ElementName='"
			<< mix->GetElement(j)->GetName() << "'), "
	    
			<< "(select IsotopeId from Isotope where IsotopeName='"
			<< mix->GetElement(j)->GetIsotope(k)->GetName() << "'), "

			<< mix->GetElement(j)->GetRelativeAbundance(k) 

			<< " from dual where not exists "
			<< "(select * from IsotopesInElement where "
			<< "ElementId=(select ElementId from Element where ElementName='"
			<< mix->GetElement(j)->GetName() << "') && "

			<< "IsotopeId=(select IsotopeId from Isotope where IsotopeName='"
			<< mix->GetElement(j)->GetIsotope(k)->GetName() << "') && "

			<< "RelativeAbundance="
			<< mix->GetElement(j)->GetRelativeAbundance(k) << ");\n";
	}
      }
    }
    outFile_volume.close();
  }
  

  for(unsigned int i=0; i < vlist->GetEntriesFast(); i++)
  {
    TGeoVolume *vol = (TGeoVolume*)vlist->UncheckedAt( i ); 

    double voldaughters=0.0;
    for(unsigned int j=0; j< vol->GetNdaughters(); j++)
    {
      voldaughters= voldaughters+ vol->GetNode(j)->GetVolume()->Capacity();
    }

    // Print current ID
    if( (i+1) % 100 == 0 || i+1==vlist->GetEntriesFast())
      printf("Volume(%d) is Done..(%d%%)\n", i+1, vol->GetNumber()*100/vlist->GetEntriesFast() );

    
    // ** Volume Table
    //+-----------------------+-------------+------+-----+---------+-------+
    //| Field                 | Type        | Null | Key | Default | Extra |
    //+-----------------------+-------------+------+-----+---------+-------+
    //| id                    | int(11)     | NO   | PRI | NULL    |       |
    //| VolumeName            | varchar(50) | YES  |     | NULL    |       |
    //| ShapeName             | varchar(50) | YES  |     | NULL    |       |
    //| ShapeType             | varchar(20) | YES  |     | NULL    |       |
    //| MaterialId            | int(11)     | YES  |     | NULL    |       |
    //| Ndaughters            | int(11)     | YES  |     | NULL    |       |
    //| Capacity              | double      | YES  |     | NULL    |       |
    //| TotalCapacityChildren | double      | YES  |     | NULL    |       |
    //| VolumeType            | varchar(10) | YES  |     | NULL    |       |
    //| Geometry              | varchar(50) | YES  |     | NULL    |       |
    //+-----------------------+-------------+------+-----+---------+-------+

    // Insert the data to Volume Table
    outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
    outFile_volume<< setprecision(15)
		  << "insert into Volume "
		  << "(id, VolumeName, ShapeName, MaterialId, Ndaughters, Capacity, TotalCapacityChildren, VolumeType, Geometry) "
		  << "values ("
		  << id << ", "
		  << "'"<< vol->GetName() << "', "
		  << "'"<< vol->GetShape()->GetName() << "', "

		  << "(select MaterialId from Material where MaterialName='"
		  << vol->GetMaterial()->GetName() << "'), "

		  << vol->GetNdaughters() << ", "
		  << vol->GetShape()->Capacity() << ", "
		  << voldaughters << ", "
		  << "'real', " 
		  << "'"<< geometry << "');\n";
    outFile_volume.close();


    // Insert Daughter information
    if( vol->GetNdaughters() )
    {
      Daughter_Volume( vol, id, m_id );
    }


    // Get Volume Information
    if( strcmp("TGeoBBox", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Box' where id="<< id << ";\n";
      outFile_volume.close();
      id= Box_Shape( id, (TGeoBBox*)vol->GetShape() );
    }
    else if( strcmp("TGeoTubeSeg", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume.close();
      id= Tube_Shape( id, (TGeoTubeSeg*)vol->GetShape() );
    }    
   else if( strcmp("TGeoTube", vol->GetShape()->ClassName() ) == 0 )
    {
      std::cerr << "Found a tube\n";
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume.close();
      id= Tube_ShapeN( id, (TGeoTube*)vol->GetShape() );
    }    
   else if( strcmp("TGeoConeSeg", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Cone' where id="<< id << ";\n";
      outFile_volume.close();
      id= Cone_Shape( id, (TGeoConeSeg*)vol->GetShape() );
    }    
   else if( strcmp("TGeoCone", vol->GetShape()->ClassName() ) == 0 )
    {
      std::cerr << "Found a cone\n";
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Cone' where id="<< id << ";\n";
      outFile_volume.close();
      id= Cone_ShapeN( id, (TGeoCone*)vol->GetShape() );
    }    
    else if( strcmp("TGeoTorus", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Torus' where id="<< id << ";\n";
      outFile_volume.close();
      id= Torus_Shape( id, (TGeoTorus*)vol->GetShape() );
    }    
    else if( strcmp("TGeoArb8", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Arb8' where id="<< id << ";\n";
      outFile_volume.close();
      id= Arb8_Shape( id, (TGeoArb8*)vol->GetShape() );
    }
    else if( strcmp("TGeoTrap", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trap' where id="<< id << ";\n";
      outFile_volume.close();
      id= Trap_Shape( id, (TGeoTrap*)vol->GetShape() );
    }   
    else if( strcmp("TGeoPcon", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polycone' where id="<< id << ";\n";
      outFile_volume.close();
      id= Polycone_Shape( id, (TGeoPcon*)vol->GetShape() );
    }
    else if( strcmp("TGeoPgon", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polyhedra' where id="<< id << ";\n";
      outFile_volume.close();
      id= Polyhedra_Shape( id, (TGeoPgon*)vol->GetShape() );
    }
    else if( strcmp("TGeoTrd2", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trd' where id="<< id << ";\n";
      outFile_volume.close();
      id= Trd_Shape( id, (TGeoTrd2*)vol->GetShape() );
    }
    else if( strcmp("TGeoCompositeShape", vol->GetShape()->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='BooleanSolid' where id="<< id << ";\n";
      outFile_volume.close();
      int depth= 1;
      id= BooleanSolid_Shape( id, (TGeoCompositeShape*)vol->GetShape(), 
			      depth, m_id, geometry );
    }
    else
    {
      printf("'%s' Table doesn't exist\n\n", vol->GetShape()->ClassName() );
      vol->GetShape()->InspectShape();
      //  exit(0);
    }
  }
}


int Box_Shape(int id, TGeoBBox *box)
{ 
  // ** Box Table
  //+-------+---------+------+-----+---------+-------+
  //| Field | Type    | Null | Key | Default | Extra |
  //+-------+---------+------+-----+---------+-------+
  //| id    | int(11) | NO   | PRI | NULL    |       |
  //| x     | double  | YES  |     | NULL    |       |
  //| y     | double  | YES  |     | NULL    |       |
  //| z     | double  | YES  |     | NULL    |       |
  //+-------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Box values ("
		<< id << ", "
		<< box->GetDX() << ", "
		<< box->GetDY() << ", "
		<< box->GetDZ() << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Tube_Shape(int id, TGeoTubeSeg *tube)
{ 
  // ** Tube Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| rmin     | double  | YES  |     | NULL    |       |
  //| rmax     | double  | YES  |     | NULL    |       |
  //| z        | double  | YES  |     | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  double startphi= tube->GetPhi1();
  double deltaphi= tube->GetPhi2()- tube->GetPhi1();
  if( (startphi+deltaphi) > 360.0 )
    startphi= startphi- 360.0;

  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Tube values ("
		<< id << ", "
		<< tube->GetRmin() << ", "
		<< tube->GetRmax() << ", "
		<< tube->GetDz() << ", "
		<< startphi << ", "
		<< deltaphi << ");\n";
  outFile_volume.close();

  id++;
  return id;
}


int Tube_ShapeN(int id, TGeoTube *tube)
{ 
  // ** Tube Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| rmin     | double  | YES  |     | NULL    |       |
  //| rmax     | double  | YES  |     | NULL    |       |
  //| z        | double  | YES  |     | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  double startphi= 0.;
  double deltaphi= 360.;
 
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Tube values ("
		<< id << ", "
		<< tube->GetRmin() << ", "
		<< tube->GetRmax() << ", "
		<< tube->GetDz() << ", "
		<< startphi << ", "
		<< deltaphi << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Cone_Shape(int id, TGeoConeSeg *cone)
{ 
  // ** Cone Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| rmin1    | double  | YES  |     | NULL    |       |
  //| rmax1    | double  | YES  |     | NULL    |       |
  //| rmin2    | double  | YES  |     | NULL    |       |
  //| rmax2    | double  | YES  |     | NULL    |       |
  //| z        | double  | YES  |     | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  double startphi= cone->GetPhi1();
  double deltaphi= cone->GetPhi2()- cone->GetPhi1();
  if( (startphi+deltaphi) > 360.0 )
    startphi= startphi- 360.0;

  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Cone values ("
		<< id << ", "
		<< cone->GetRmin1() << ", "
		<< cone->GetRmax1() << ", "
		<< cone->GetRmin2() << ", "
		<< cone->GetRmax2() << ", "
		<< cone->GetDz() << ", "
		<< startphi << ", "
		<< deltaphi << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Cone_ShapeN(int id, TGeoCone*cone)
{ 
  // ** Cone Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| rmin1    | double  | YES  |     | NULL    |       |
  //| rmax1    | double  | YES  |     | NULL    |       |
  //| rmin2    | double  | YES  |     | NULL    |       |
  //| rmax2    | double  | YES  |     | NULL    |       |
  //| z        | double  | YES  |     | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  double startphi= 0.;
  double deltaphi= 360.;

  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Cone values ("
		<< id << ", "
		<< cone->GetRmin1() << ", "
		<< cone->GetRmax1() << ", "
		<< cone->GetRmin2() << ", "
		<< cone->GetRmax2() << ", "
		<< cone->GetDz() << ", "
		<< startphi << ", "
		<< deltaphi << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Torus_Shape(int id, TGeoTorus *torus)
{ 
  // ** Torus Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| rmin     | double  | YES  |     | NULL    |       |
  //| rmax     | double  | YES  |     | NULL    |       |
  //| rtor     | double  | YES  |     | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Torus values ("
		<< id << ", "
		<< torus->GetRmin() << ", "
		<< torus->GetRmax() << ", "
		<< torus->GetR() << ", "
		<< torus->GetPhi1() << ", "
		<< torus->GetDphi() << ");\n";
 outFile_volume.close();

 id++;
 return id;
}

int Arb8_Shape(int id, TGeoArb8 *arb8)
{
  // ** Arb8 Table
  //+-------+---------+------+-----+---------+-------+
  //| Field | Type    | Null | Key | Default | Extra |
  //+-------+---------+------+-----+---------+-------+
  //| id    | int(11) | NO   | PRI | NULL    |       |
  //| v1x   | double  | YES  |     | NULL    |       |
  //| v1y   | double  | YES  |     | NULL    |       |
  //| v2x   | double  | YES  |     | NULL    |       |
  //| v2y   | double  | YES  |     | NULL    |       |
  //| v3x   | double  | YES  |     | NULL    |       |
  //| v3y   | double  | YES  |     | NULL    |       |
  //| v4x   | double  | YES  |     | NULL    |       |
  //| v4y   | double  | YES  |     | NULL    |       |
  //| v5x   | double  | YES  |     | NULL    |       |
  //| v5y   | double  | YES  |     | NULL    |       |
  //| v6x   | double  | YES  |     | NULL    |       |
  //| v6y   | double  | YES  |     | NULL    |       |
  //| v7x   | double  | YES  |     | NULL    |       |
  //| v7y   | double  | YES  |     | NULL    |       |
  //| v8x   | double  | YES  |     | NULL    |       |
  //| v8y   | double  | YES  |     | NULL    |       |
  //| dz    | double  | YES  |     | NULL    |       |
  //+-------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Arb8 values ("
		<< id << ", "
		<< arb8->GetVertices()[0] << ", "
		<< arb8->GetVertices()[1] << ", "
		<< arb8->GetVertices()[2] << ", "
		<< arb8->GetVertices()[3] << ", "
		<< arb8->GetVertices()[4] << ", "
		<< arb8->GetVertices()[5] << ", "
		<< arb8->GetVertices()[6] << ", "
		<< arb8->GetVertices()[7] << ", "
		<< arb8->GetVertices()[8] << ", "
		<< arb8->GetVertices()[9] << ", "
		<< arb8->GetVertices()[10] << ", "
		<< arb8->GetVertices()[11] << ", "
		<< arb8->GetVertices()[12] << ", "
		<< arb8->GetVertices()[13] << ", "
		<< arb8->GetVertices()[14] << ", "
		<< arb8->GetVertices()[15] << ", "
		<< arb8->GetDz() << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Trap_Shape(int id, TGeoTrap *trap)
{
  // ** Trap Table
  //+--------+---------+------+-----+---------+-------+
  //| Field  | Type    | Null | Key | Default | Extra |
  //+--------+---------+------+-----+---------+-------+
  //| id     | int(11) | NO   | PRI | NULL    |       |
  //| z      | double  | YES  |     | NULL    |       |
  //| theta  | double  | YES  |     | NULL    |       |
  //| phi    | double  | YES  |     | NULL    |       |
  //| y1     | double  | YES  |     | NULL    |       |
  //| x1     | double  | YES  |     | NULL    |       |
  //| x2     | double  | YES  |     | NULL    |       |
  //| alpha1 | double  | YES  |     | NULL    |       |
  //| y2     | double  | YES  |     | NULL    |       |
  //| x3     | double  | YES  |     | NULL    |       |
  //| x4     | double  | YES  |     | NULL    |       |
  //| alpha2 | double  | YES  |     | NULL    |       |
  //+--------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Trap values ("
		<< id << ", "
		<< trap->GetDz() << ", "
		<< trap->GetTheta() << ", "
		<< trap->GetPhi() << ", "
		<< trap->GetH1() << ", "
		<< trap->GetBl1() << ", "
		<< trap->GetTl1() << ", "
		<< trap->GetAlpha1() << ", "
		<< trap->GetH2() << ", "
		<< trap->GetBl2() << ", "
		<< trap->GetTl2() << ", "
		<< trap->GetAlpha2() << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int Polycone_Shape(int id, TGeoPcon *pcon)
{
  // ** Polycone Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //| zplane   | int(11) | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Polycone values ("
		<< id << ", "
		<< pcon->GetPhi1() << ", "
		<< pcon->GetDphi() << ", "
		<< pcon->GetNz() << ");\n";

  
  // ** Polycone_plane Table
  //+-----------+---------+------+-----+---------+-------+
  //| Field     | Type    | Null | Key | Default | Extra |
  //+-----------+---------+------+-----+---------+-------+
  //| id        | int(11) | YES  |     | NULL    |       |
  //| plane_num | int(11) | YES  |     | NULL    |       |
  //| rmin      | double  | YES  |     | NULL    |       |
  //| rmax      | double  | YES  |     | NULL    |       |
  //| z         | double  | YES  |     | NULL    |       |
  //+-----------+---------+------+-----+---------+-------+

  for(unsigned int i= 1; i<= pcon->GetNz(); i++)
  {
    outFile_volume<< setprecision(15)
		  << "insert into Polycone_plane values ("
		  << id << ", "
		  << i << ", "
		  << pcon->GetRmin(i-1) << ", "
		  << pcon->GetRmax(i-1) << ", "
		  << pcon->GetZ(i-1) << ");\n";
  }
  outFile_volume.close();

  id++;
  return id;
}

int Polyhedra_Shape(int id, TGeoPgon *pgon)
{
  // ** Polyhedra Table
  //+----------+---------+------+-----+---------+-------+
  //| Field    | Type    | Null | Key | Default | Extra |
  //+----------+---------+------+-----+---------+-------+
  //| id       | int(11) | NO   | PRI | NULL    |       |
  //| startphi | double  | YES  |     | NULL    |       |
  //| deltaphi | double  | YES  |     | NULL    |       |
  //| numsides | int(11) | YES  |     | NULL    |       |
  //| zplane   | int(11) | YES  |     | NULL    |       |
  //+----------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Polyhedra values ("
		<< id << ", "
		<< pgon->GetPhi1() << ", "
		<< pgon->GetDphi() << ", "
		<< pgon->GetNedges() << ", "
		<< pgon->GetNz() << ");\n";

  
  // ** Polyhedra_plane Table
  //+-----------+---------+------+-----+---------+-------+
  //| Field     | Type    | Null | Key | Default | Extra |
  //+-----------+---------+------+-----+---------+-------+
  //| id        | int(11) | YES  |     | NULL    |       |
  //| plane_num | int(11) | YES  |     | NULL    |       |
  //| rmin      | double  | YES  |     | NULL    |       |
  //| rmax      | double  | YES  |     | NULL    |       |
  //| z         | double  | YES  |     | NULL    |       |
  //+-----------+---------+------+-----+---------+-------+

  for(unsigned int i= 1; i<= pgon->GetNz(); i++)
  {
    outFile_volume<< setprecision(15)
		  << "insert into Polyhedra_plane values ("
		  << id << ", "
		  << i << ", "
		  << pgon->GetRmin(i-1) << ", "
		  << pgon->GetRmax(i-1) << ", "
		  << pgon->GetZ(i-1) << ");\n";
  }
  outFile_volume.close();

  id++;
  return id;
}

int Trd_Shape(int id, TGeoTrd2 *trd2)
{ 
  // ** Trd Table
  //+-------+---------+------+-----+---------+-------+
  //| Field | Type    | Null | Key | Default | Extra |
  //+-------+---------+------+-----+---------+-------+
  //| id    | int(11) | NO   | PRI | NULL    |       |
  //| x1    | double  | YES  |     | NULL    |       |
  //| x2    | double  | YES  |     | NULL    |       |
  //| y1    | double  | YES  |     | NULL    |       |
  //| y2    | double  | YES  |     | NULL    |       |
  //| z     | double  | YES  |     | NULL    |       |
  //+-------+---------+------+-----+---------+-------+

  ofstream outFile_volume;
  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
  outFile_volume<< setprecision(15)
		<< "insert into Trd values ("
		<< id << ", "
		<< trd2->GetDx1() << ", "
		<< trd2->GetDx2() << ", "
		<< trd2->GetDy1() << ", "
		<< trd2->GetDy2() << ", "
		<< trd2->GetDz() << ");\n";
  outFile_volume.close();

  id++;
  return id;
}

int BooleanSolid_Shape(int id, TGeoCompositeShape *comp, int depth, int *m_id, char* geometry)
{
  TGeoCompositeShape* comp_l= (TGeoCompositeShape*)( comp->GetBoolNode()->GetLeftShape() );
  TGeoCompositeShape* comp_r= (TGeoCompositeShape*)( comp->GetBoolNode()->GetRightShape() );
  int current_shape_id= id;

  ofstream outFile_volume;

  // ** BooleanSolid Table
  //+----------------+-------------+------+-----+---------+-------+
  //| Field          | Type        | Null | Key | Default | Extra |
  //+----------------+-------------+------+-----+---------+-------+
  //| id             | int(11)     | NO   | PRI | NULL    |       |
  //| LeftShapeID    | int(11)     | YES  |     | NULL    |       |
  //| RightShapeID   | int(11)     | YES  |     | NULL    |       |
  //| R_mID          | int(11)     | YES  |     | NULL    |       |
  //| LeftShapeType  | varchar(20) | YES  |     | NULL    |       |
  //| RightShapeType | varchar(20) | YES  |     | NULL    |       |
  //| Operator       | varchar(20) | YES  |     | NULL    |       |
  //| Depth          | int(11)     | YES  |     | NULL    |       |
  //+----------------+-------------+------+-----+---------+-------+

  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);

  char Operator[20];
  if( comp->GetBoolNode()->GetBooleanOperator() == 0 )
    strcpy(Operator, "Union"); 
  else if( comp->GetBoolNode()->GetBooleanOperator() == 1 )
    strcpy(Operator, "Intersection"); 
  else if( comp->GetBoolNode()->GetBooleanOperator() == 2 )
    strcpy(Operator, "Subtraction"); 

   
  // Insert BooleanSolid
  outFile_volume<< "insert into BooleanSolid "
		<< "(id, Operator, Depth) values ("
		<< id << ", "
		<< "'" << Operator << "', "
		<< depth << ");\n";
  id++;

  // Update Left Solid ID and TransformationID
  outFile_volume<< "update BooleanSolid set "
		<< "LeftShapeID=" << id 
		<< " where id=" << current_shape_id << ";\n";

  // Insert Left Solid at volume Table
  outFile_volume<< "insert into Volume (id, ShapeName, VolumeType, Geometry) values ("
		<< id << ", "
		<< "'"<< comp_l->GetName() << "', "
		<< "'virtual', " 
		<< "'"<< geometry << "');\n";
  
  outFile_volume.close();


  if( comp_l->IsComposite() )
  {
    outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
    outFile_volume<< "update Volume set ShapeType='BooleanSolid' where id="<< id << ";\n";
    outFile_volume<< "update BooleanSolid set LeftShapeType='BooleanSolid' "
		  << "where id=" << current_shape_id << ";\n";
    outFile_volume.close();
    id= BooleanSolid_Shape( id, comp_l, depth+1, m_id, geometry );
  }
  else
  {
    // Get Primitive Shape of Left Solid
    if( strcmp("TGeoBBox", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Box' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Box' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Box_Shape( id, (TGeoBBox*)comp_l );
    }
    else if( strcmp("TGeoTubeSeg", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Tube' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Tube_Shape( id, (TGeoTubeSeg*)comp_l );
    }   
   else if( strcmp("TGeoTube", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Tube' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Tube_ShapeN( id, (TGeoTube*)comp_l );
    }   
  
   else if( strcmp("TGeoConeSeg", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Cone' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Cone' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Cone_Shape( id, (TGeoConeSeg*)comp_l );
    }
    else if( strcmp("TGeoTorus", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Torus' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Torus' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Torus_Shape( id, (TGeoTorus*)comp_l );
    }
    else if( strcmp("TGeoArb8", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Arb8' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Arb8' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Arb8_Shape( id, (TGeoArb8*)comp_l );
    }
    else if( strcmp("TGeoTrap", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trap' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Trap' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Trap_Shape( id, (TGeoTrap*)comp_l );
    }
    else if( strcmp("TGeoPcon", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polycone' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Polycone' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Polycone_Shape( id, (TGeoPcon*)comp_l );
    }
    else if( strcmp("TGeoPgon", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polyhedra' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Polyhedra' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Polyhedra_Shape( id, (TGeoPgon*)comp_l );
    }
    else if( strcmp("TGeoTrd2", comp_l->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trd' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set LeftShapeType='Trd' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Trd_Shape( id, (TGeoTrd2*)comp_l );
    }
    else
    {
      printf("'%s' Table doesn't exist while updating composite \n\n", comp_l->ClassName() );
      exit(0);
    }
  }


  outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);

  // Update Right Solid ID and TransformationID
  outFile_volume<< "update BooleanSolid set "
		<< "RightShapeID=" << id 
		<< ", R_mID=" << *m_id 
		<< " where id=" << current_shape_id << ";\n";

  // Insert Placement information about Right Solid
  outFile_volume<< setprecision(15)
		<< "insert into Transformation values ("
    		<< *m_id << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetTranslation())[0] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetTranslation())[1] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetTranslation())[2] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[0] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[1] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[2] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[3] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[4] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[5] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[6] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[7] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetRotationMatrix())[8] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetScale())[0] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetScale())[1] << ", "
		<< (comp->GetBoolNode()->GetRightMatrix()->GetScale())[2] << ");\n";
  (*m_id)++;

  // Insert Right Solid at volume Table
  outFile_volume<< "insert into Volume (id, ShapeName, VolumeType, Geometry) values ("
		<< id << ", "
		<< "'"<< comp_r->GetName() << "', "
		<< "'virtual', " 
		<< "'"<< geometry << "');\n";

  outFile_volume.close();


  if( comp_r->IsComposite() )
  {
    outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
    outFile_volume<< "update Volume set ShapeType='BooleanSolid' where id="<< id << ";\n";
    outFile_volume<< "update BooleanSolid set RightShapeType='BooleanSolid' "
		  << "where id=" << current_shape_id << ";\n";
    outFile_volume.close();
    id= BooleanSolid_Shape( id, comp_r, depth+1, m_id, geometry );
  }
  else
  {
    // Get Primitive Shape of Right Shape
    if( strcmp("TGeoBBox", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Box' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Box' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Box_Shape( id, (TGeoBBox*)comp_r );
    }
    else if( strcmp("TGeoTubeSeg", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Tube' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Tube_Shape( id, (TGeoTubeSeg*)comp_r );
    }
    else if( strcmp("TGeoTube", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Tube' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Tube' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Tube_ShapeN( id, (TGeoTube*)comp_r );
    }
    else if( strcmp("TGeoConeSeg", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Cone' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Cone' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Cone_Shape( id, (TGeoConeSeg*)comp_r );
    }
    else if( strcmp("TGeoTorus", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Torus' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Torus' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Torus_Shape( id, (TGeoTorus*)comp_r );
    } 
    else if( strcmp("TGeoArb8", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Arb8' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Arb8' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Arb8_Shape( id, (TGeoArb8*)comp_r );
    }
    else if( strcmp("TGeoTrap", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trap' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Trap' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Trap_Shape( id, (TGeoTrap*)comp_r );
    }
    else if( strcmp("TGeoPcon", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polycone' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Polycone' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Polycone_Shape( id, (TGeoPcon*)comp_r );
    }
    else if( strcmp("TGeoPgon", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Polyhedra' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Polyhedra' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Polyhedra_Shape( id, (TGeoPgon*)comp_r );
    }
    else if( strcmp("TGeoTrd2", comp_r->ClassName() ) == 0 )
    {
      outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);
      outFile_volume<< "update Volume set ShapeType='Trd' where id="<< id << ";\n";
      outFile_volume<< "update BooleanSolid set RightShapeType='Trd' "
		    << "where id=" << current_shape_id << ";\n";
      outFile_volume.close();
      id= Trd_Shape( id, (TGeoTrd2*)comp_r );
    }
    else
    {
      printf("'%s' Table doesn't exist\n\n", comp_r->ClassName() );
      exit(0);
    }
  }

  return id;
}


void Daughter_Volume(TGeoVolume *vol, int parent_id, int *m_id)
{
  ofstream outFile_volume;

  for(unsigned int i= 0; i< vol->GetNdaughters(); i++)
  {
    outFile_volume.open("output.sql", std::ios_base::out | std::ios_base::app);

    // ** Daughter_Volume Table
    //+------------------+---------+------+-----+---------+-------+
    //| Field            | Type    | Null | Key | Default | Extra |
    //+------------------+---------+------+-----+---------+-------+
    //| id               | int(11) | YES  |     | NULL    |       |
    //| DaughterVolumeId | int(11) | YES  |     | NULL    |       |
    //| mID              | int(11) | YES  |     | NULL    |       |
    //+------------------+---------+------+-----+---------+-------+

    outFile_volume<< "insert into Daughter_Volume select "
		  << parent_id << ", "
		  << "id, "
		  << *m_id 
		  << " from Volume where VolumeName='"
		  << vol->GetNode(i)->GetVolume()->GetName() << "';\n";


    // ** Transformation Table
    //+---------------+---------+------+-----+---------+-------+
    //| Field         | Type    | Null | Key | Default | Extra |
    //+---------------+---------+------+-----+---------+-------+
    //| mID           | int(11) | NO   | PRI | 0       |       |
    //| Translation_1 | double  | YES  |     | NULL    |       |
    //| Translation_2 | double  | YES  |     | NULL    |       |
    //| Translation_3 | double  | YES  |     | NULL    |       |
    //| Rotation_1    | double  | YES  |     | NULL    |       |
    //| Rotation_2    | double  | YES  |     | NULL    |       |
    //| Rotation_3    | double  | YES  |     | NULL    |       |
    //| Rotation_4    | double  | YES  |     | NULL    |       |
    //| Rotation_5    | double  | YES  |     | NULL    |       |
    //| Rotation_6    | double  | YES  |     | NULL    |       |
    //| Rotation_7    | double  | YES  |     | NULL    |       |
    //| Rotation_8    | double  | YES  |     | NULL    |       |
    //| Rotation_9    | double  | YES  |     | NULL    |       |
    //| Scale_1       | double  | YES  |     | NULL    |       |
    //| Scale_2       | double  | YES  |     | NULL    |       |
    //| Scale_3       | double  | YES  |     | NULL    |       |
    //+---------------+---------+------+-----+---------+-------+

    // Insert Placement information about Daughter volume
    outFile_volume<< setprecision(15)
		  << "insert into Transformation values ("
      		  << *m_id << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetTranslation())[0] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetTranslation())[1] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetTranslation())[2] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[0] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[1] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[2] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[3] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[4] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[5] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[6] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[7] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetRotationMatrix())[8] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetScale())[0] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetScale())[1] << ", "
		  << (vol->GetNode(i)->GetMatrix()->GetScale())[2] << ");\n";
    (*m_id)++;

    outFile_volume.close();
  }
}
