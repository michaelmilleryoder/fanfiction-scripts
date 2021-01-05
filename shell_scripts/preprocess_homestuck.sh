fandom=homestuck
#python /projects/fanfiction_lgbtq/scripts/filter_fics.py $fandom complete_en_1k-50k
python /projects/fanfiction_lgbtq/scripts/preprocess_fics.py /projects/AO3Scraper/output/ao3_${fandom}_text /data/fanfiction_ao3/${fandom}/complete_en_1k-50k/ --update-existing --num-copy-workers 30 --num-tok-workers 30
