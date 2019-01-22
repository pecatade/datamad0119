CREATE DATABASE IF NOT EXISTS lab_mysql;
USE lab_mysql;
CREATE TABLE IF NOT EXISTS Cars (
  `ID` INT,
  `VIN` VARCHAR(45),
  `Manufacturer` VARCHAR(45),
  `Model` VARCHAR(45),
  `Year` INT,
  `Color` VARCHAR(45),
  PRIMARY KEY (`ID`));
CREATE TABLE IF NOT EXISTS Customers (
  `ID` INT,
  `Customer ID` INT,
  `Name` VARCHAR(45),
  `Phone` VARCHAR(20),
  `Email` VARCHAR(45),
  `Address` VARCHAR(45),
  `City` VARCHAR(45),
  `State/Province` VARCHAR(45),
  `Country` VARCHAR(45),
  `Postal` INT,
  PRIMARY KEY (`ID`));
CREATE TABLE IF NOT EXISTS Salespersons (
  `ID` INT,
  `Staff ID` INT,
  `Name` VARCHAR(45),
  `Store` VARCHAR(45),
  PRIMARY KEY (`ID`));
CREATE TABLE IF NOT EXISTS Invoices (
  `ID` INT,
  `Invoice number` INT,
  `Date` DATE,
  `Car` INT,
  `Customer` INT,
  `Sales person` INT,
  PRIMARY KEY (`ID`));
CREATE TABLE IF NOT EXISTS Sale (
  `Salespersons_ID` INT,
  `Invoices_ID` INT,
  `Cars_ID` INT,
  `Customers_ID` INT NOT NULL,
  PRIMARY KEY (`Salespersons_ID`, `Invoices_ID`, `Cars_ID`, `Customers_ID`),
  INDEX `fk_Sale_Salespersons1_idx` (`Salespersons_ID` ASC) VISIBLE,
  INDEX `fk_Sale_Invoices1_idx` (`Invoices_ID` ASC) VISIBLE,
  INDEX `fk_Sale_Cars1_idx` (`Cars_ID` ASC) VISIBLE,
  INDEX `fk_Sale_Customers1_idx` (`Customers_ID` ASC) VISIBLE,
  CONSTRAINT `fk_Sale_Salespersons1`
    FOREIGN KEY (`Salespersons_ID`)
    REFERENCES `lab_mysql`.`Salespersons` (`ID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Sale_Invoices1`
    FOREIGN KEY (`Invoices_ID`)
    REFERENCES `lab_mysql`.`Invoices` (`ID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Sale_Cars1`
    FOREIGN KEY (`Cars_ID`)
    REFERENCES `lab_mysql`.`Cars` (`ID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Sale_Customers1`
    FOREIGN KEY (`Customers_ID`)
    REFERENCES `lab_mysql`.`Customers` (`ID`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
INSERT INTO `lab_mysql`.`Cars` (`ID`,`VIN`,`Manufacturer`,`Model`,`Year`,`Color`) 
VALUES 
('0','3K096I98581DHSNUP','Volkswagen','Tiguan','2019','Blue'),
('1','ZM8G7BEUQZ97IH46V','Peugeot','Rifter','2019','Red'),
('2','RKXVNNIHLVVZOUB4M','Ford','Fusion','2018','White'),
('3','HKNDGS7CU31E9Z7JW','Toyota','RAV4','2018','Silver'),
('4','DAM41UDN3CHU2WVF6','Volvo','V60','2019','Gray'),
('5','DAM41UDN3CHU2WVF6','Volvo','V60 Cross Country','2019','Gray');
INSERT INTO `lab_mysql`.`customers` (`ID`,`Customer ID`,`Name`,`Phone`,`Email`,`Address`,`City`,`State/Province`,`Country`,`Postal`)
VALUES
('0','10001','Pablo Picasso','+34 636 17 63 82','-','Paseo de la Chopera, 14','Madrid','Madrid','Spain','28045'),
('1','20001','Abraham Lincoln','+1 305 907 7086','-','120 SW 8th St','Miami','Florida','United States','33130'),
('2','30001','Napoléon Bonaparte','+33 1 79 75 40 00','-','40 Rue du Colisée','Paris','Île-de-France','France','75008');
INSERT INTO `lab_mysql`.`Salespersons` (`ID`,`Staff ID`,`Name`,`Store`)
VALUES
('0','00001','Petey Cruiser','Madrid'),
('1','00002','Anna Sthesia','Barcelona'),
('2','00003','Paul Molive','Berlin'),
('3','00004','Gail Forcewind','Paris'),
('4','00005','Paige Turner','Mimia'),
('5','00006','Bob Frapples','Mexico City'),
('6','00007','Walter Melon','Amsterdam'),
('7','00008','Shonda Leer','São Paulo');
INSERT INTO `lab_mysql`.`Invoices` (`ID`,`Invoice number`,`Date`,`Car`,`Customer`,`Sales person`)
VALUES
('0','852399038','2018-08-22','0','1','3'),
('1','731166526','2018-12-31','3','0','5'),
('2','271135104','2019-01-22','2','2','7');