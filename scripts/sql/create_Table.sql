-- MySQL dump 10.13  Distrib 5.1.73, for redhat-linux-gnu (x86_64)
--
-- Host: localhost    Database: root
-- ------------------------------------------------------
-- Server version	5.1.73

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Arb8`
--

DROP TABLE IF EXISTS `Arb8`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Arb8` (
  `id` int(11) NOT NULL,
  `v1x` double DEFAULT NULL,
  `v1y` double DEFAULT NULL,
  `v2x` double DEFAULT NULL,
  `v2y` double DEFAULT NULL,
  `v3x` double DEFAULT NULL,
  `v3y` double DEFAULT NULL,
  `v4x` double DEFAULT NULL,
  `v4y` double DEFAULT NULL,
  `v5x` double DEFAULT NULL,
  `v5y` double DEFAULT NULL,
  `v6x` double DEFAULT NULL,
  `v6y` double DEFAULT NULL,
  `v7x` double DEFAULT NULL,
  `v7y` double DEFAULT NULL,
  `v8x` double DEFAULT NULL,
  `v8y` double DEFAULT NULL,
  `dz` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Arb8`
--

LOCK TABLES `Arb8` WRITE;
/*!40000 ALTER TABLE `Arb8` DISABLE KEYS */;
/*!40000 ALTER TABLE `Arb8` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `BooleanSolid`
--

DROP TABLE IF EXISTS `BooleanSolid`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `BooleanSolid` (
  `id` int(11) NOT NULL,
  `LeftShapeID` int(11) DEFAULT NULL,
  `RightShapeID` int(11) DEFAULT NULL,
  `R_mID` int(11) DEFAULT NULL,
  `LeftShapeType` varchar(20) DEFAULT NULL,
  `RightShapeType` varchar(20) DEFAULT NULL,
  `Operator` varchar(20) DEFAULT NULL,
  `Depth` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `BooleanSolid`
--

LOCK TABLES `BooleanSolid` WRITE;
/*!40000 ALTER TABLE `BooleanSolid` DISABLE KEYS */;
/*!40000 ALTER TABLE `BooleanSolid` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Box`
--

DROP TABLE IF EXISTS `Box`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Box` (
  `id` int(11) NOT NULL,
  `x` double DEFAULT NULL,
  `y` double DEFAULT NULL,
  `z` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Box`
--

LOCK TABLES `Box` WRITE;
/*!40000 ALTER TABLE `Box` DISABLE KEYS */;
/*!40000 ALTER TABLE `Box` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Cone`
--

DROP TABLE IF EXISTS `Cone`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Cone` (
  `id` int(11) NOT NULL,
  `rmin1` double DEFAULT NULL,
  `rmax1` double DEFAULT NULL,
  `rmin2` double DEFAULT NULL,
  `rmax2` double DEFAULT NULL,
  `z` double DEFAULT NULL,
  `startphi` double DEFAULT NULL,
  `deltaphi` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Cone`
--

LOCK TABLES `Cone` WRITE;
/*!40000 ALTER TABLE `Cone` DISABLE KEYS */;
/*!40000 ALTER TABLE `Cone` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Daughter_Volume`
--

DROP TABLE IF EXISTS `Daughter_Volume`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Daughter_Volume` (
  `id` int(11) DEFAULT NULL,
  `DaughterVolumeId` int(11) DEFAULT NULL,
  `mID` int(11) DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Daughter_Volume`
--

LOCK TABLES `Daughter_Volume` WRITE;
/*!40000 ALTER TABLE `Daughter_Volume` DISABLE KEYS */;
/*!40000 ALTER TABLE `Daughter_Volume` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Element`
--

DROP TABLE IF EXISTS `Element`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Element` (
  `ElementId` int(11) NOT NULL AUTO_INCREMENT,
  `ElementName` varchar(50) DEFAULT NULL,
  `A` double DEFAULT NULL,
  `N` int(11) DEFAULT NULL,
  `Z` int(11) DEFAULT NULL,
  `Nisotopes` int(11) DEFAULT NULL,
  PRIMARY KEY (`ElementId`),
  UNIQUE KEY `ElementName` (`ElementName`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Element`
--

LOCK TABLES `Element` WRITE;
/*!40000 ALTER TABLE `Element` DISABLE KEYS */;
/*!40000 ALTER TABLE `Element` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ElementsInMaterial`
--

DROP TABLE IF EXISTS `ElementsInMaterial`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ElementsInMaterial` (
  `MaterialId` int(11) DEFAULT NULL,
  `ElementId` int(11) DEFAULT NULL,
  `Weight` double DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ElementsInMaterial`
--

LOCK TABLES `ElementsInMaterial` WRITE;
/*!40000 ALTER TABLE `ElementsInMaterial` DISABLE KEYS */;
/*!40000 ALTER TABLE `ElementsInMaterial` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Isotope`
--

DROP TABLE IF EXISTS `Isotope`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Isotope` (
  `IsotopeId` int(11) NOT NULL AUTO_INCREMENT,
  `IsotopeName` varchar(50) DEFAULT NULL,
  `A` double DEFAULT NULL,
  `N` int(11) DEFAULT NULL,
  `Z` int(11) DEFAULT NULL,
  PRIMARY KEY (`IsotopeId`),
  UNIQUE KEY `IsotopeName` (`IsotopeName`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Isotope`
--

LOCK TABLES `Isotope` WRITE;
/*!40000 ALTER TABLE `Isotope` DISABLE KEYS */;
/*!40000 ALTER TABLE `Isotope` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `IsotopesInElement`
--

DROP TABLE IF EXISTS `IsotopesInElement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IsotopesInElement` (
  `ElementId` int(11) DEFAULT NULL,
  `IsotopeId` int(11) DEFAULT NULL,
  `RelativeAbundance` double DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `IsotopesInElement`
--

LOCK TABLES `IsotopesInElement` WRITE;
/*!40000 ALTER TABLE `IsotopesInElement` DISABLE KEYS */;
/*!40000 ALTER TABLE `IsotopesInElement` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Material`
--

DROP TABLE IF EXISTS `Material`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Material` (
  `MaterialId` int(11) NOT NULL AUTO_INCREMENT,
  `MaterialName` varchar(50) DEFAULT NULL,
  `A` double DEFAULT NULL,
  `Z` double DEFAULT NULL,
  `Density` double DEFAULT NULL,
  `Pressure` double DEFAULT NULL,
  `Temperature` double DEFAULT NULL,
  `RadLen` double DEFAULT NULL,
  `IntLen` double DEFAULT NULL,
  `Nelements` int(11) DEFAULT NULL,
  `IsMixture` int(11) DEFAULT NULL,
  PRIMARY KEY (`MaterialId`),
  UNIQUE KEY `MaterialName` (`MaterialName`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Material`
--

LOCK TABLES `Material` WRITE;
/*!40000 ALTER TABLE `Material` DISABLE KEYS */;
/*!40000 ALTER TABLE `Material` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Polycone`
--

DROP TABLE IF EXISTS `Polycone`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Polycone` (
  `id` int(11) NOT NULL,
  `startphi` double DEFAULT NULL,
  `deltaphi` double DEFAULT NULL,
  `zplane` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Polycone`
--

LOCK TABLES `Polycone` WRITE;
/*!40000 ALTER TABLE `Polycone` DISABLE KEYS */;
/*!40000 ALTER TABLE `Polycone` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Polycone_plane`
--

DROP TABLE IF EXISTS `Polycone_plane`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Polycone_plane` (
  `id` int(11) DEFAULT NULL,
  `plane_num` int(11) DEFAULT NULL,
  `rmin` double DEFAULT NULL,
  `rmax` double DEFAULT NULL,
  `z` double DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Polycone_plane`
--

LOCK TABLES `Polycone_plane` WRITE;
/*!40000 ALTER TABLE `Polycone_plane` DISABLE KEYS */;
/*!40000 ALTER TABLE `Polycone_plane` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Polyhedra`
--

DROP TABLE IF EXISTS `Polyhedra`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Polyhedra` (
  `id` int(11) NOT NULL,
  `startphi` double DEFAULT NULL,
  `deltaphi` double DEFAULT NULL,
  `numsides` int(11) DEFAULT NULL,
  `zplane` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Polyhedra`
--

LOCK TABLES `Polyhedra` WRITE;
/*!40000 ALTER TABLE `Polyhedra` DISABLE KEYS */;
/*!40000 ALTER TABLE `Polyhedra` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Polyhedra_plane`
--

DROP TABLE IF EXISTS `Polyhedra_plane`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Polyhedra_plane` (
  `id` int(11) DEFAULT NULL,
  `plane_num` int(11) DEFAULT NULL,
  `rmin` double DEFAULT NULL,
  `rmax` double DEFAULT NULL,
  `z` double DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Polyhedra_plane`
--

LOCK TABLES `Polyhedra_plane` WRITE;
/*!40000 ALTER TABLE `Polyhedra_plane` DISABLE KEYS */;
/*!40000 ALTER TABLE `Polyhedra_plane` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `StressTest`
--

DROP TABLE IF EXISTS `StressTest`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `StressTest` (
  `Date` datetime DEFAULT NULL,
  `VolumeID` int(11) DEFAULT NULL,
  `UorV` varchar(15) DEFAULT NULL,
  `errCode` int(11) DEFAULT NULL,
  `errInsidePoint` int(11) DEFAULT NULL,
  `errOutsidePoint` int(11) DEFAULT NULL,
  `errSurfacePoint` int(11) DEFAULT NULL,
  `errSafetyFromInside` int(11) DEFAULT NULL,
  `errSafetyFromOutside` int(11) DEFAULT NULL,
  `errShapeDistances` int(11) DEFAULT NULL,
  `errAccuracyDistanceToIn` int(11) DEFAULT NULL,
  `errFarAwayPoint` int(11) DEFAULT NULL,
  `errShapeNormal` int(11) DEFAULT NULL,
  `errIntegration` int(11) DEFAULT NULL,
  `errXRayProfile` int(11) DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `StressTest`
--

LOCK TABLES `StressTest` WRITE;
/*!40000 ALTER TABLE `StressTest` DISABLE KEYS */;
/*!40000 ALTER TABLE `StressTest` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `TestedVolume`
--

DROP TABLE IF EXISTS `TestedVolume`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `TestedVolume` (
  `Date` datetime DEFAULT NULL,
  `VolumeID` int(11) NOT NULL,
  `HostName` varchar(50) DEFAULT NULL,
  `gccVersion` varchar(50) DEFAULT NULL,
  `ROOTVersion` varchar(50) DEFAULT NULL,
  `GEANT4Version` varchar(50) DEFAULT NULL,
  `errorFound` int(11) DEFAULT NULL,
  `NumberOfParticle` int(11) DEFAULT NULL,
  `Tolerance` double DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `TestedVolume`
--

LOCK TABLES `TestedVolume` WRITE;
/*!40000 ALTER TABLE `TestedVolume` DISABLE KEYS */;
/*!40000 ALTER TABLE `TestedVolume` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Torus`
--

DROP TABLE IF EXISTS `Torus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Torus` (
  `id` int(11) NOT NULL,
  `rmin` double DEFAULT NULL,
  `rmax` double DEFAULT NULL,
  `rtor` double DEFAULT NULL,
  `startphi` double DEFAULT NULL,
  `deltaphi` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Torus`
--

LOCK TABLES `Torus` WRITE;
/*!40000 ALTER TABLE `Torus` DISABLE KEYS */;
/*!40000 ALTER TABLE `Torus` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Transformation`
--

DROP TABLE IF EXISTS `Transformation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Transformation` (
  `mID` int(11) NOT NULL DEFAULT '0',
  `Translation_1` double DEFAULT NULL,
  `Translation_2` double DEFAULT NULL,
  `Translation_3` double DEFAULT NULL,
  `Rotation_1` double DEFAULT NULL,
  `Rotation_2` double DEFAULT NULL,
  `Rotation_3` double DEFAULT NULL,
  `Rotation_4` double DEFAULT NULL,
  `Rotation_5` double DEFAULT NULL,
  `Rotation_6` double DEFAULT NULL,
  `Rotation_7` double DEFAULT NULL,
  `Rotation_8` double DEFAULT NULL,
  `Rotation_9` double DEFAULT NULL,
  `Scale_1` double DEFAULT NULL,
  `Scale_2` double DEFAULT NULL,
  `Scale_3` double DEFAULT NULL,
  PRIMARY KEY (`mID`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Transformation`
--

LOCK TABLES `Transformation` WRITE;
/*!40000 ALTER TABLE `Transformation` DISABLE KEYS */;
/*!40000 ALTER TABLE `Transformation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Trap`
--

DROP TABLE IF EXISTS `Trap`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Trap` (
  `id` int(11) NOT NULL,
  `z` double DEFAULT NULL,
  `theta` double DEFAULT NULL,
  `phi` double DEFAULT NULL,
  `y1` double DEFAULT NULL,
  `x1` double DEFAULT NULL,
  `x2` double DEFAULT NULL,
  `alpha1` double DEFAULT NULL,
  `y2` double DEFAULT NULL,
  `x3` double DEFAULT NULL,
  `x4` double DEFAULT NULL,
  `alpha2` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Trap`
--

LOCK TABLES `Trap` WRITE;
/*!40000 ALTER TABLE `Trap` DISABLE KEYS */;
/*!40000 ALTER TABLE `Trap` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Trd`
--

DROP TABLE IF EXISTS `Trd`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Trd` (
  `id` int(11) NOT NULL,
  `x1` double DEFAULT NULL,
  `x2` double DEFAULT NULL,
  `y1` double DEFAULT NULL,
  `y2` double DEFAULT NULL,
  `z` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Trd`
--

LOCK TABLES `Trd` WRITE;
/*!40000 ALTER TABLE `Trd` DISABLE KEYS */;
/*!40000 ALTER TABLE `Trd` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Tube`
--

DROP TABLE IF EXISTS `Tube`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Tube` (
  `id` int(11) NOT NULL,
  `rmin` double DEFAULT NULL,
  `rmax` double DEFAULT NULL,
  `z` double DEFAULT NULL,
  `startphi` double DEFAULT NULL,
  `deltaphi` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Tube`
--

LOCK TABLES `Tube` WRITE;
/*!40000 ALTER TABLE `Tube` DISABLE KEYS */;
/*!40000 ALTER TABLE `Tube` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Volume`
--

DROP TABLE IF EXISTS `Volume`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Volume` (
  `id` int(11) NOT NULL,
  `VolumeName` varchar(50) DEFAULT NULL,
  `ShapeName` varchar(50) DEFAULT NULL,
  `ShapeType` varchar(20) DEFAULT NULL,
  `MaterialId` int(11) DEFAULT NULL,
  `Ndaughters` int(11) DEFAULT NULL,
  `Capacity` double DEFAULT NULL,
  `TotalCapacityChildren` double DEFAULT NULL,
  `VolumeType` varchar(10) DEFAULT NULL,
  `Geometry` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Volume`
--

LOCK TABLES `Volume` WRITE;
/*!40000 ALTER TABLE `Volume` DISABLE KEYS */;
/*!40000 ALTER TABLE `Volume` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2015-02-13 19:06:17
