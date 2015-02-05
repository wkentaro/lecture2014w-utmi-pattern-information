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

(setq xtr1 (load-data "../data/Train1.txt"))
(print xtr1)
