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
  (setq data nil)
  (dolist (x raw)
    (setq data (append data (list (split x)))))
  data)
