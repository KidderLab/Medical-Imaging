# Medical_Imaging

**# CancerTFs**************

**CellNet setup file (script_setupCellNet.R)** <br>

**Set your library path so that it points to the correct platform and annotation libraries** <br>

.libPaths("~/myprog/cellnetr/packages") <br>
library("cellnetr") <br>

**human** <br>
library("org.Hs.eg.db") <br>

**hgu133plus2** <br>
library("hgu133plus2.db"); <br>
library("hgu133plus2cdf"); <br>

**set up path for the CellNet objects containing the classifiers, GRNs** <br>
path_CN_obj<-"~/myprog/cellnetr/training_data/"; <br>

**change to reflect your platform** <br>
myPlatform<-"hgu133plus2" <br>

mydir<-"~/myprog/cellnetr/final_cellnet/CellNet-master/" <br>
source( paste(mydir, "CellNet_sourceme.R", sep='') ); <br>

**this sources all of R files** <br>
utils_sourceRs(mydir); <br>


**CellNet main script (script_maincellnetr.R)** <br>

source("script_setupCellNet.R") <br>

mydate<-utils_myDate(); <br>

cellnetfiledir <- "~/myprog/cellnetr" <br>
fileName <- "final_stall_11102022" <br>

cat("# Reading stAll file ...\n") <br>
stQuery<-expr_readSampTab(paste(fileName,".csv",sep="")); <br>
stQuery<-geo_fixNames(stQuery); <br>
stAll<-utils_loadObject(paste("stQuery_",fileName,".R",sep="")); <br>


**THIS WILL LOAD THE CEL FILES and make raw gene expression measurements** <br>

library(affy); <br>
cat("# Reading all Cel files ...\n") <br>
expAll<-Norm_cleanPropRaw(stAll, "hgu133plus2") <br>

**select samples for GRN reconstruction** <br>

cat("# Creating stGRN ...\n") <br>
stGRN<-sample_profiles_grn(stAll, minNum=84); <br>

cat("# Creating expGRN ...\n") <br>
expR<-expAll[,rownames(stGRN)]; <br>
expGRN<-Norm_quantNorm(expR); <br>

**Determine latest TR annotation** <br>
hTFs<-find_tfs("Hs"); <br>

**Correlations and Zscores for GRN** <br>
corrX<-grn_corr_round(expGRN); <br>
hTFs<-intersect(hTFs, rownames(expGRN)); <br>
zscs<-grn_zscores(corrX, hTFs); <br>

**EMPIRICALLY DETERMINED by comparison to 3 GOLD STANDARDS** <br>
zthresh<-6; <br>

ctGRNs<-cn_grnDoRock(stGRN, expGRN, zscs, corrX, "general", dLevelGK="description6", zThresh=zthresh); <br>

**Make the complete CellNet object using all data** <br>
cnProc<-cn_make_processor(expAll, stAll, ctGRNs, dLevel="description1", classWeight=TRUE, exprWeight=TRUE); <br>
fname<-paste(“cnProc_",fileName, mydate, ".R", sep=''); <br>
save(cnProc, file=fname); <br>


**Rainbow plot (script_cellnet_rainbowPlot.R)** <br>

source("script_setupCellNet.R") <br>

.libPaths("~/myprog/cellnetr/packages") <br>

library("cellnetr") <br>
library("hgu133plus2.db"); <br>
library("hgu133plus2cdf"); <br>
library("randomForest") <br>
library(gplots) <br>
library(ggplot2) <br>


path_CN_obj<-"~/cellnet_2022/cancer/output_full/"; <br>
outputfileName<- outputfileName <br>
targetCT<- targetCT <br>
csvFile<- csvFile <br>
cnproc_data<- cnproc_data <br>

myPlatform<-"hgu133plus2" <br>
cName<-"description1" <br>

stQuery<-expr_readSampTab(paste("normal_cells_query/",csvFile,sep="")); <br>
stQuery<-geo_fixNames(stQuery); <br>
cnObjName<-switch(myPlatform,hgu133plus2 = paste(path_CN_obj,cnproc_data ,sep='')); <br>
expQuery<-Norm_cleanPropRaw(stQuery, myPlatform); <br>
cnProc<-utils_loadObject(cnObjName); <br>
tmpAns<-cn_apply(expQuery, stQuery, cnProc, dLevelQuery=cName); <br>
tfScores<-cn_nis_all(tmpAns, cnProc, targetCT); <br>

pdf(paste(outputfileName,".pdf",sep="")) <br>

cn_hmClass(tmpAns,isBig = TRUE); <br>

**Gene regulatory network status of starting cell type (esc) GRN** <br>
cn_barplot_grnSing(tmpAns, cnProc, targetCT, c(targetCT), bOrder=NULL); <br>

**Network influence score of HSPC GRN transcriptional regulators** <br>
cn_plotnis(tfScores[[targetCT]], limit=15); <br>

  scoresDF<-tfScores[[targetCT]] <br>
  xmeans<-apply(scoresDF, 2, mean); <br>
  worst<-which.min(xmeans); <br>
  tfs<-rownames(scoresDF)[order(scoresDF[,worst], decreasing=F)]; <br>
  scoresDF<-scoresDF[tfs,]; <br>
  topTF<-rownames(scoresDF); <br>
  write.table(topTF,paste("TFs_",outputfileName,".txt",sep=""),row.names=F,quote=F) <br>

plot(mp_rainbowPlot(cnProc[['expTrain']],cnProc[['stTrain']],topTF[i], dLevel="description1")) <br>
dev.off() <br>


**GRN_ROCS (script_cellnet_grn_rocs.sh)** <br>

source("script_setupCellNet.R") <br>

cellnetfiledir <- "~/myprog/cellnetr" <br>

filename="final_stall_11102022" <br>
mydate<-utils_myDate(); <br>

stQuery<-expr_readSampTab(paste(filename,".csv",sep="")); <br>

stQuery<-geo_fixNames(stQuery); <br>
stAll <- utils_loadObject(paste("stQuery",filename,".R",sep="")) <br>

library(affy); <br>

**load expall** <br>
load(paste("expAll_",filename, mydate,".R",sep="")); <br>
expProp <- expAll <br>

**load stGRN** <br>
load(paste(“stGRN_",filename, mydate,".R",sep="")); <br>

**load expGRN** <br>
load(paste("expGRN_",filename, mydate,".R",sep="")); <br>

**load tfs** <br>
load(paste("tfs_",filename, mydate,".R",sep="")); <br>

**load corrx htd zscs** <br>
load(paste("tmpforctGRN_",filename, mydate,".rda",sep="")); <br>

zthresh<-6; <br>

load(paste("ctGRNs_",filename, mydate,".R",sep="")); <br>

**Grn report** <br>
grn_report(ctGRNs); <br>
fname<-paste("GRN_report_",filename,".pdf", sep=''); <br>
ggsave(file=fname, width=8.5, height=11); <br>
dev.copy(pdf, file=fname, width=8.5, height=11); <br>
dev.off() <br>

**select classification training and validation data** <br>
stList<-samp_for_class(stAll, prop=0.5, dLevel="description1") <br>
lapply(stList, nrow) <br>
stVal<-stList[['stVal']]; <br>
stTrain<-stList[['stTrain']]; <br>
expTrain<-expProp[,rownames(stTrain)]; <br>
library(randomForest) <br>
system.time(classifiers<-cn_makeRFs(expTrain, stTrain, ctGRNs$ctGRNs$geneLists)); <br>
expVal<-expProp[,rownames(stVal)] <br>
ansVal<-cn_classify(classifiers, expVal, ctGRNs[[3]]$geneLists) <br>
assessed<-cn_classAssess(ansVal,stVal, classLevels="description2", resolution=0.01); <br>

plot_class_rocs(assessed) <br>

fname<-paste("Rocs_final_",filename,".pdf",sep='');
ggsave(file=fname, width=6, height=7);
dev.off();

