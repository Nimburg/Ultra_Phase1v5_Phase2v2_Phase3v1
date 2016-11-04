library(animation)
library(plotrix)

# Load the data. Source: US Census Bureau
tag_relevence <- read.csv("data/Tag_July_relevence.csv", stringsAsFactors=FALSE)

# Temporary image directory
ani.options(outdir = paste(getwd(), "/images", sep=""))

saveGIF( 
for (day in unique(tag_relevence$date)) {
  tag_day <- subset(tag_relevence, date==day)
  day_relev_1 <- tag_day$relev_1
  day_relev_2 <- tag_day$relev_2

  par(cex=1.0)
  par(
      mar=pyramid.plot(day_relev_1, day_relev_2, 
      top.labels=c("Trump", "", "Hillary"), 
      labels=tag_day$tag, main=day, lxcol="#A097CC", rxcol="#EDBFBE", 
      xlim=c(20,20), 
      gap=10,unit="relevence scaled between 0 and 10")
     )
}, 
  movie.name = "tag_july_relevence.gif", interval=2, 
  nmax=50, ani.width=600, ani.height=400)










