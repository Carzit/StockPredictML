python data.py --data "data\stocks\ALOT.csv" --save "data\preprocessed_data.csv"

python fixed_partition.py --data "data\preprocessed_data.csv"

python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 5 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 10 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 15 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 20 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 25 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 30 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 40 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 50 --test_len 1
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 100 --test_len 1

python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 5 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 10 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 15 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 20 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 25 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 30 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 40 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 50 --test_len 2
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 100 --test_len 2

python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 5 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 10 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 15 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 20 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 25 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 30 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 40 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 50 --test_len 3
python fixed_window.py --data "data\preprocessed_data.csv" --log "log\FixedWindow.txt" --window_len 100 --test_len 3

python cumulative_window.py --data "data\preprocessed_data.csv" --log "log\CumulativeWindow.txt" --test_len 1
python cumulative_window.py --data "data\preprocessed_data.csv" --log "log\CumulativeWindow.txt" --test_len 2
python cumulative_window.py --data "data\preprocessed_data.csv" --log "log\CumulativeWindow.txt" --test_len 3
