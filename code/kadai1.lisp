#!/usr/bin/env eus

(defun split (str)
  (read-from-string (concatenate string "(" str ")")))


(defun load-data (fname)
  ; raw data
  (with-open-file (stream fname)
    (setq raw nil)
    (do ((line (read-line stream nil)
              (read-line stream nil)))
      ((null line))
      (setq raw (append raw (list line)))))
  ; make dataset
  (setq data (make-matrix (length raw) 2))
  (setq i 0)
  (dolist (x raw)
    (setf (matrix-row data i) (split x))
    (incf i))
  data)

(defun get-labels (len val)
  (setq ret nil)
  (dotimes (i len) (setq ret (cons val ret)))
  ret)

; train data
(setq xtr1 (load-data "../data/Train1.txt"))
(setq ytr1 (get-labels (send xtr1 :get-val 'dim0) 0))
(setq xtr2 (load-data "../data/Train2.txt"))
(setq ytr2 (get-labels (send xtr2 :get-val 'dim0) 1))
; test data
(setq xts1 (load-data "../data/Test1.txt"))
(setq yts1 (get-labels (send xts1 :get-val 'dim0) 0))
(setq xts2 (load-data "../data/Test2.txt"))
(setq yts2 (get-labels (send xts2 :get-val 'dim0) 1))

