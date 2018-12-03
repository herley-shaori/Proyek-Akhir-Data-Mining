from fitur_makanan import Fitur
from klas import Klas
# from contour_feature import CF

# Menghasilkan hog.csv. Panggil untuk pertama kali.
# fitur_makan = Fitur()

kelas = Klas()
# kelas.svm_trial()
# kelas.pca_dt_trial()
# kelas.pca_svm_trial()
# kelas.xgb_trial(with_pca=False)
# kelas.random_forest_trial(with_pca=False)
# kelas.extra_trees_trial(with_pca=False)
kelas.ada_boost_trial(with_pca=False)

# cf = CF()