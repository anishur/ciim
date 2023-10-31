CREATE DATABASE  IF NOT EXISTS `db_urban_waste_by_year` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `db_urban_waste_by_year`;
-- MySQL dump 10.13  Distrib 8.0.34, for Win64 (x86_64)
--
-- Host: localhost    Database: db_urban_waste_by_year
-- ------------------------------------------------------
-- Server version	8.0.34

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `waste_collected_by_year`
--

DROP TABLE IF EXISTS `waste_collected_by_year`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `waste_collected_by_year` (
  `id` int NOT NULL AUTO_INCREMENT,
  `year` int NOT NULL,
  `region` varchar(50) DEFAULT NULL,
  `total` int DEFAULT NULL,
  `papel` int DEFAULT NULL,
  `plastico` int DEFAULT NULL,
  `metal` int DEFAULT NULL,
  `vidro` int DEFAULT NULL,
  `madeira` int DEFAULT NULL,
  `equipamentos` int DEFAULT NULL,
  `pilhas` int DEFAULT NULL,
  `oleos_alimentares` int DEFAULT NULL,
  `outros` int DEFAULT NULL,
  `recolha_indiferenciada` int DEFAULT NULL,
  `recolha_selectiva` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `waste_collected_by_year`
--

LOCK TABLES `waste_collected_by_year` WRITE;
/*!40000 ALTER TABLE `waste_collected_by_year` DISABLE KEYS */;
INSERT INTO `waste_collected_by_year` VALUES (1,2017,'CIM Coimbra',16352,5629,3432,31,6919,81,103,1,0,156,161652,25257),(2,2017,'CIM Viseu',6477,2123,1234,65,2536,47,150,0,0,322,89047,9801),(3,2017,'CIM Beiras',6682,2532,1227,148,2175,257,228,3,0,114,76017,9729),(4,2018,'CIM Coimbra',19087,7210,3643,38,7814,30,130,1,1,220,168396,28160),(5,2018,'CIM Viseu',7312,2485,1428,89,2703,54,177,1,2,374,90585,11213),(6,2018,'CIM Beiras',7076,2725,1340,198,2284,138,259,3,1,128,78333,9846),(7,2019,'CIM Coimbra',24099,8634,4497,43,9661,88,162,2,5,206,164876,34456),(8,2019,'CIM Viseu',8763,2976,1761,88,3219,75,195,0,12,437,89601,13022),(9,2019,'CIM Beiras',8181,2974,1654,201,2480,101,275,2,3,162,77222,10511),(10,2020,'CIM Coimbra',25497,9398,5334,54,8931,67,189,3,4,231,164158,36424),(11,2020,'CIM Viseu',10963,3755,2392,123,3895,111,234,1,18,435,90577,16975),(12,2020,'CIM Beiras',9316,3363,1993,227,2699,168,402,4,2,176,77536,11890);
/*!40000 ALTER TABLE `waste_collected_by_year` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-10-30 17:40:33