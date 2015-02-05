#!/usr/bin/env eus

(defun split (str)
  (read-from-string (concatenate string "(" str ")")))

(with-open-file (stream "../data/Train1.txt")
  (setq raw nil)
  (do ((line (read-line stream nil)
             (read-line stream nil)))
    ((null line))
    (setq raw (append raw (list line)))))

(setq data nil)
(dolist (x raw)
  (setq data (append data (list (split x)))))
(print data)