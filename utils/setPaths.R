# Default Sloan Cluster
if(Sys.info()['sysname']=="Linux"){
  DATADIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/"
  RAWDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/raw/"
  TEMPDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/temp/"
  PROCDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/processed/"
  OUTDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/"
  PLOTSDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/plots/"
  TABLESDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/tables/"
}else{
  switch(Sys.info()["user"],
         "chang.2590" = {
           DIR <- "D:/Dropbox/project/muni_bonds/"
           DATADIR <- "D:/Dropbox/project/muni_bonds/data"
           RAWDIR <- "D:/Dropbox/project/muni_bonds/data/raw/"
           TEMPDIR <- "D:/Dropbox/project/muni_bonds/data/temp/"
           PROCDIR <- "D:/Dropbox/project/muni_bonds/data/processed/"
           OUTDIR <- "D:/Dropbox/project/muni_bonds/outputs/"
           PLOTSDIR <- "D:/Dropbox/project/muni_bonds/outputs/plots/"
           TABLESDIR <- "D:/Dropbox/project/muni_bonds/outputs/tables/"
         },
         "User" = {
           DIR <- "D:/Dropbox/project/muni_bonds/"
           DATADIR <- "F:/Dropbox (Personal)/project/muni_bonds/data"
           RAWDIR <- "F:/Dropbox (Personal)/project/muni_bonds/data/raw/"
           TEMPDIR <- "F:/Dropbox (Personal)/project/muni_bonds/data/temp/"
           PROCDIR <- "F:/Dropbox (Personal)/project/muni_bonds/data/processed/"
           OUTDIR <- "F:/Dropbox (Personal)/project/muni_bonds/outputs/"
           PLOTSDIR <- "F:/Dropbox (Personal)/project/muni_bonds/outputs/plots/"
           TABLESDIR <- "F:/Dropbox (Personal)/project/muni_bonds/outputs/tables/"
         }
         )
}
