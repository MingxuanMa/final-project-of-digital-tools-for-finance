USE dtft_db;

CREATE TABLE riskfreerate (
	Dates VARCHAR(10),
    Yield VARCHAR(6)
    );
    
load data infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/riskfreerate.csv'
into table riskfreerate
fields terminated by ',' optionally enclosed by '"' escaped by '"'
lines terminated by '\r\n'
ignore 1 lines
;

CREATE TABLE sp500 (
Dates VARCHAR(12),
AUN VARCHAR(7),
AALUW VARCHAR(7),
AAPUN VARCHAR(7),
AAPLUW VARCHAR(7),
ABBVUN VARCHAR(7),
ABCUN VARCHAR(7),
ABMDUW VARCHAR(7),
ABTUN VARCHAR(7),
ACNUN VARCHAR(7),
ADBEUW VARCHAR(7),
ADIUW VARCHAR(7),
ADMUN VARCHAR(7),
ADPUW VARCHAR(7),
ADSKUW VARCHAR(7),
AEEUN VARCHAR(7),
AEPUW VARCHAR(7),
AESUN VARCHAR(7),
AFLUN VARCHAR(7),
AIGUN VARCHAR(7),
AIVUN VARCHAR(7),
AIZUN VARCHAR(7),
AJGUN VARCHAR(7),
AKAMUW VARCHAR(7),
ALBUN VARCHAR(7),
ALGNUW VARCHAR(7),
ALKUN VARCHAR(7),
ALLUN VARCHAR(7),
ALLEUN VARCHAR(7),
ALXNUW VARCHAR(7),
AMATUW VARCHAR(7),
AMCRUN VARCHAR(7),
AMDUW VARCHAR(7),
AMEUN VARCHAR(7),
AMGNUW VARCHAR(7),
AMPUN VARCHAR(7),
AMTUN VARCHAR(7),
AMZNUW VARCHAR(7),
ANETUN VARCHAR(7),
ANSSUW VARCHAR(7),
ANTMUN VARCHAR(7),
AONUN VARCHAR(7),
AOSUN VARCHAR(7),
APAUW VARCHAR(7),
APDUN VARCHAR(7),
APHUN VARCHAR(7),
APTVUN VARCHAR(7),
AREUN VARCHAR(7),
ATOUN VARCHAR(7),
ATVIUW VARCHAR(7),
AVBUN VARCHAR(7),
AVGOUW VARCHAR(7),
AVYUN VARCHAR(7),
AWKUN VARCHAR(7),
AXPUN VARCHAR(7),
AZOUN VARCHAR(7),
BAUN VARCHAR(7),
BACUN VARCHAR(7),
BAXUN VARCHAR(7),
BBYUN VARCHAR(7),
BDXUN VARCHAR(7),
BENUN VARCHAR(7),
BFBUN VARCHAR(7),
BIIBUW VARCHAR(7),
BIOUN VARCHAR(7),
BKUN VARCHAR(7),
BKNGUW VARCHAR(7),
BKRUN VARCHAR(7),
BLKUN VARCHAR(7),
BLLUN VARCHAR(7),
BMYUN VARCHAR(7),
BRUN VARCHAR(7),
BRKBUN VARCHAR(7),
BSXUN VARCHAR(7),
BWAUN VARCHAR(7),
BXPUN VARCHAR(7),
CUN VARCHAR(7),
CAGUN VARCHAR(7),
CAHUN VARCHAR(7),
CARRUN VARCHAR(7),
CATUN VARCHAR(7),
CBUN VARCHAR(7),
CBOEUF VARCHAR(7),
CBREUN VARCHAR(7),
CCIUN VARCHAR(7),
CCLUN VARCHAR(7),
CDNSUW VARCHAR(7),
CDWUW VARCHAR(7),
CEUN VARCHAR(7),
CERNUW VARCHAR(7),
CFUN VARCHAR(7),
CFGUN VARCHAR(7),
CHDUN VARCHAR(7),
CHRWUW VARCHAR(7),
CHTRUW VARCHAR(7),
CIUN VARCHAR(7),
CINFUW VARCHAR(7),
CLUN VARCHAR(7),
CLXUN VARCHAR(7),
CMAUN VARCHAR(7),
CMCSAUW VARCHAR(7),
CMEUW VARCHAR(7),
CMGUN VARCHAR(7),
CMIUN VARCHAR(7),
CMSUN VARCHAR(7),
CNCUN VARCHAR(7),
CNPUN VARCHAR(7),
COFUN VARCHAR(7),
COGUN VARCHAR(7),
COOUN VARCHAR(7),
COPUN VARCHAR(7),
COSTUW VARCHAR(7),
CPBUN VARCHAR(7),
CPRTUW VARCHAR(7),
CRMUN VARCHAR(7),
CSCOUW VARCHAR(7),
CSXUW VARCHAR(7),
CTASUW VARCHAR(7),
CTLTUN VARCHAR(7),
CTSHUW VARCHAR(7),
CTVAUN VARCHAR(7),
CTXSUW VARCHAR(7),
CVSUN VARCHAR(7),
CVXUN VARCHAR(7),
CXOUN VARCHAR(7),
DUN VARCHAR(7),
DALUN VARCHAR(7),
DDUN VARCHAR(7),
DEUN VARCHAR(7),
DFSUN VARCHAR(7),
DGUN VARCHAR(7),
DGXUN VARCHAR(7),
DHIUN VARCHAR(7),
DHRUN VARCHAR(7),
DISUN VARCHAR(7),
DISCAUW VARCHAR(7),
DISCKUW VARCHAR(7),
DISHUW VARCHAR(7),
DLRUN VARCHAR(7),
DLTRUW VARCHAR(7),
DOVUN VARCHAR(7),
DOWUN VARCHAR(7),
DPZUN VARCHAR(7),
DREUN VARCHAR(7),
DRIUN VARCHAR(7),
DTEUN VARCHAR(7),
DUKUN VARCHAR(7),
DVAUN VARCHAR(7),
DVNUN VARCHAR(7),
DXCUN VARCHAR(7),
DXCMUW VARCHAR(7),
EAUW VARCHAR(7),
EBAYUW VARCHAR(7),
ECLUN VARCHAR(7),
EDUN VARCHAR(7),
EFXUN VARCHAR(7),
EIXUN VARCHAR(7),
ELUN VARCHAR(7),
EMNUN VARCHAR(7),
EMRUN VARCHAR(7),
EOGUN VARCHAR(7),
EQIXUW VARCHAR(7),
EQRUN VARCHAR(7),
ESUN VARCHAR(7),
ESSUN VARCHAR(7),
ETNUN VARCHAR(7),
ETRUN VARCHAR(7),
ETSYUW VARCHAR(7),
EVRGUN VARCHAR(7),
EWUN VARCHAR(7),
EXCUW VARCHAR(7),
EXPDUW VARCHAR(7),
EXPEUW VARCHAR(7),
EXRUN VARCHAR(7),
FUN VARCHAR(7),
FANGUW VARCHAR(7),
FASTUW VARCHAR(7),
FBUW VARCHAR(7),
FBHSUN VARCHAR(7),
FCXUN VARCHAR(7),
FDXUN VARCHAR(7),
FEUN VARCHAR(7),
FFIVUW VARCHAR(7),
FISUN VARCHAR(7),
FISVUW VARCHAR(7),
FITBUW VARCHAR(7),
FLIRUW VARCHAR(7),
FLSUN VARCHAR(7),
FLTUN VARCHAR(7),
FMCUN VARCHAR(7),
FOXUW VARCHAR(7),
FOXAUW VARCHAR(7),
FRCUN VARCHAR(7),
FRTUN VARCHAR(7),
FTIUN VARCHAR(7),
FTNTUW VARCHAR(7),
FTVUN VARCHAR(7),
GDUN VARCHAR(7),
GEUN VARCHAR(7),
GILDUW VARCHAR(7),
GISUN VARCHAR(7),
GLUN VARCHAR(7),
GLWUN VARCHAR(7),
GMUN VARCHAR(7),
GOOGUW VARCHAR(7),
GOOGLUW VARCHAR(7),
GPCUN VARCHAR(7),
GPNUN VARCHAR(7),
GPSUN VARCHAR(7),
GRMNUW VARCHAR(7),
GSUN VARCHAR(7),
GWWUN VARCHAR(7),
HALUN VARCHAR(7),
HASUW VARCHAR(7),
HBANUW VARCHAR(7),
HBIUN VARCHAR(7),
HCAUN VARCHAR(7),
HDUN VARCHAR(7),
HESUN VARCHAR(7),
HFCUN VARCHAR(7),
HIGUN VARCHAR(7),
HIIUN VARCHAR(7),
HLTUN VARCHAR(7),
HOLXUW VARCHAR(7),
HONUN VARCHAR(7),
HPEUN VARCHAR(7),
HPQUN VARCHAR(7),
HRLUN VARCHAR(7),
HSICUW VARCHAR(7),
HSTUN VARCHAR(7),
HSYUN VARCHAR(7),
HUMUN VARCHAR(7),
HWMUN VARCHAR(7),
IBMUN VARCHAR(7),
ICEUN VARCHAR(7),
IDXXUW VARCHAR(7),
IEXUN VARCHAR(7),
IFFUN VARCHAR(7),
ILMNUW VARCHAR(7),
INCYUW VARCHAR(7),
INFOUN VARCHAR(7),
INTCUW VARCHAR(7),
INTUUW VARCHAR(7),
IPUN VARCHAR(7),
IPGUN VARCHAR(7),
IPGPUW VARCHAR(7),
IQVUN VARCHAR(7),
IRUN VARCHAR(7),
IRMUN VARCHAR(7),
ISRGUW VARCHAR(7),
ITUN VARCHAR(7),
ITWUN VARCHAR(7),
IVZUN VARCHAR(7),
JUN VARCHAR(7),
JBHTUW VARCHAR(7),
JCIUN VARCHAR(7),
JKHYUW VARCHAR(7),
JNJUN VARCHAR(7),
JNPRUN VARCHAR(7),
JPMUN VARCHAR(7),
KUN VARCHAR(7),
KEYUN VARCHAR(7),
KEYSUN VARCHAR(7),
KHCUW VARCHAR(7),
KIMUN VARCHAR(7),
KLACUW VARCHAR(7),
KMBUN VARCHAR(7),
KMIUN VARCHAR(7),
KMXUN VARCHAR(7),
KOUN VARCHAR(7),
KRUN VARCHAR(7),
KSUUN VARCHAR(7),
LUN VARCHAR(7),
LBUN VARCHAR(7),
LDOSUN VARCHAR(7),
LEGUN VARCHAR(7),
LENUN VARCHAR(7),
LHUN VARCHAR(7),
LHXUN VARCHAR(7),
LINUN VARCHAR(7),
LKQUW VARCHAR(7),
LLYUN VARCHAR(7),
LMTUN VARCHAR(7),
LNCUN VARCHAR(7),
LNTUW VARCHAR(7),
LOWUN VARCHAR(7),
LRCXUW VARCHAR(7),
LUMNUN VARCHAR(7),
LUVUN VARCHAR(7),
LVSUN VARCHAR(7),
LWUN VARCHAR(7),
LYBUN VARCHAR(7),
LYVUN VARCHAR(7),
MAUN VARCHAR(7),
MAAUN VARCHAR(7),
MARUW VARCHAR(7),
MASUN VARCHAR(7),
MCDUN VARCHAR(7),
MCHPUW VARCHAR(7),
MCKUN VARCHAR(7),
MCOUN VARCHAR(7),
MDLZUW VARCHAR(7),
MDTUN VARCHAR(7),
METUN VARCHAR(7),
MGMUN VARCHAR(7),
MHKUN VARCHAR(7),
MKCUN VARCHAR(7),
MKTXUW VARCHAR(7),
MLMUN VARCHAR(7),
MMCUN VARCHAR(7),
MMMUN VARCHAR(7),
MNSTUW VARCHAR(7),
MOUN VARCHAR(7),
MOSUN VARCHAR(7),
MPCUN VARCHAR(7),
MRKUN VARCHAR(7),
MROUN VARCHAR(7),
MSUN VARCHAR(7),
MSCIUN VARCHAR(7),
MSFTUW VARCHAR(7),
MSIUN VARCHAR(7),
MTBUN VARCHAR(7),
MTDUN VARCHAR(7),
MUUW VARCHAR(7),
MXIMUW VARCHAR(7),
MYLUW VARCHAR(7),
NCLHUN VARCHAR(7),
NDAQUW VARCHAR(7),
NEEUN VARCHAR(7),
NEMUN VARCHAR(7),
NFLXUW VARCHAR(7),
NIUN VARCHAR(7),
NKEUN VARCHAR(7),
NLOKUW VARCHAR(7),
NLSNUN VARCHAR(7),
NOCUN VARCHAR(7),
NOVUN VARCHAR(7),
NOWUN VARCHAR(7),
NRGUN VARCHAR(7),
NSCUN VARCHAR(7),
NTAPUW VARCHAR(7),
NTRSUW VARCHAR(7),
NUEUN VARCHAR(7),
NVDAUW VARCHAR(7),
NVRUN VARCHAR(7),
NWLUW VARCHAR(7),
NWSUW VARCHAR(7),
NWSAUW VARCHAR(7),
OUN VARCHAR(7),
ODFLUW VARCHAR(7),
OKEUN VARCHAR(7),
OMCUN VARCHAR(7),
ORCLUN VARCHAR(7),
ORLYUW VARCHAR(7),
OTISUN VARCHAR(7),
OXYUN VARCHAR(7),
PAYCUN VARCHAR(7),
PAYXUW VARCHAR(7),
PBCTUW VARCHAR(7),
PCARUW VARCHAR(7),
PEAKUN VARCHAR(7),
PEGUN VARCHAR(7),
PEPUW VARCHAR(7),
PFEUN VARCHAR(7),
PFGUW VARCHAR(7),
PGUN VARCHAR(7),
PGRUN VARCHAR(7),
PHUN VARCHAR(7),
PHMUN VARCHAR(7),
PKGUN VARCHAR(7),
PKIUN VARCHAR(7),
PLDUN VARCHAR(7),
PMUN VARCHAR(7),
PNCUN VARCHAR(7),
PNRUN VARCHAR(7),
PNWUN VARCHAR(7),
POOLUW VARCHAR(7),
PPGUN VARCHAR(7),
PPLUN VARCHAR(7),
PRGOUN VARCHAR(7),
PRUUN VARCHAR(7),
PSAUN VARCHAR(7),
PSXUN VARCHAR(7),
PVHUN VARCHAR(7),
PWRUN VARCHAR(7),
PXDUN VARCHAR(7),
PYPLUW VARCHAR(7),
QCOMUW VARCHAR(7),
QRVOUW VARCHAR(7),
RCLUN VARCHAR(7),
REUN VARCHAR(7),
REGUW VARCHAR(7),
REGNUW VARCHAR(7),
RFUN VARCHAR(7),
RHIUN VARCHAR(7),
RJFUN VARCHAR(7),
RLUN VARCHAR(7),
RMDUN VARCHAR(7),
ROKUN VARCHAR(7),
ROLUN VARCHAR(7),
ROPUN VARCHAR(7),
ROSTUW VARCHAR(7),
RSGUN VARCHAR(7),
RTXUN VARCHAR(7),
SBACUW VARCHAR(7),
SBUXUW VARCHAR(7),
SCHWUN VARCHAR(7),
SEEUN VARCHAR(7),
SHWUN VARCHAR(7),
SIVBUW VARCHAR(7),
SJMUN VARCHAR(7),
SLBUN VARCHAR(7),
SLGUN VARCHAR(7),
SNAUN VARCHAR(7),
SNPSUW VARCHAR(7),
SOUN VARCHAR(7),
SPGUN VARCHAR(7),
SPGIUN VARCHAR(7),
SREUN VARCHAR(7),
STEUN VARCHAR(7),
STTUN VARCHAR(7),
STXUW VARCHAR(7),
STZUN VARCHAR(7),
SWKUN VARCHAR(7),
SWKSUW VARCHAR(7),
SYFUN VARCHAR(7),
SYKUN VARCHAR(7),
SYYUN VARCHAR(7),
TUN VARCHAR(7),
TAPUN VARCHAR(7),
TDGUN VARCHAR(7),
TDYUN VARCHAR(7),
TELUN VARCHAR(7),
TERUW VARCHAR(7),
TFCUN VARCHAR(7),
TFXUN VARCHAR(7),
TGTUN VARCHAR(7),
TIFUN VARCHAR(7),
TJXUN VARCHAR(7),
TMOUN VARCHAR(7),
TMUSUW VARCHAR(7),
TPRUN VARCHAR(7),
TROWUW VARCHAR(7),
TRVUN VARCHAR(7),
TSCOUW VARCHAR(7),
TSNUN VARCHAR(7),
TTUN VARCHAR(7),
TTWOUW VARCHAR(7),
TWTRUN VARCHAR(7),
TXNUW VARCHAR(7),
TXTUN VARCHAR(7),
TYLUN VARCHAR(7),
UAUN VARCHAR(7),
UAAUN VARCHAR(7),
UALUW VARCHAR(7),
UDRUN VARCHAR(7),
UHSUN VARCHAR(7),
ULTAUW VARCHAR(7),
UNHUN VARCHAR(7),
UNMUN VARCHAR(7),
UNPUN VARCHAR(7),
UPSUN VARCHAR(7),
URIUN VARCHAR(7),
USBUN VARCHAR(7),
VUN VARCHAR(7),
VARUN VARCHAR(7),
VFCUN VARCHAR(7),
VIACUW VARCHAR(7),
VLOUN VARCHAR(7),
VMCUN VARCHAR(7),
VNOUN VARCHAR(7),
VNTUN VARCHAR(7),
VRSKUW VARCHAR(7),
VRSNUW VARCHAR(7),
VRTXUW VARCHAR(7),
VTRUN VARCHAR(7),
VZUN VARCHAR(7),
WABUN VARCHAR(7),
WATUN VARCHAR(7),
WBAUW VARCHAR(7),
WDCUW VARCHAR(7),
WECUN VARCHAR(7),
WELLUN VARCHAR(7),
WFCUN VARCHAR(7),
WHRUN VARCHAR(7),
WLTWUW VARCHAR(7),
WMUN VARCHAR(7),
WMBUN VARCHAR(7),
WMTUN VARCHAR(7),
WRBUN VARCHAR(7),
WRKUN VARCHAR(7),
WSTUN VARCHAR(7),
WUUN VARCHAR(7),
WYUN VARCHAR(7),
WYNNUW VARCHAR(7),
XELUW VARCHAR(7),
XLNXUW VARCHAR(7),
XOMUN VARCHAR(7),
XRAYUW VARCHAR(7),
XRXUN VARCHAR(7),
XYLUN VARCHAR(7),
YUMUN VARCHAR(7),
ZBHUN VARCHAR(7),
ZBRAUW VARCHAR(7),
ZIONUW VARCHAR(7),
ZTSUN VARCHAR(7)
    )ENGINE=InnoDB CHARACTER SET latin1;
load data infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/sp500.csv'
ignore into table sp500
fields terminated by ',' optionally enclosed by '"' escaped by '"'
lines terminated by '\r\n'
ignore 1 lines;    