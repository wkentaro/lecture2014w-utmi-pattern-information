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



(defun get-vector (len val)
  (setq ret nil)
  (dotimes (i len) (setq ret (cons val ret)))
  (setq ret (coerce ret vector))
  ret)


; (defun get-vector (len val)
;   (setq ret (make-matrix len 1))
;   (dotimes (i len) (setf (matrix-row ret i) (list val)))
;   ret)


(defclass lms
  :super propertied-object
  :slots (eta iterations w))
(defmethod lms
  (:init (_eta _iterations)
         (setq eta _eta)
         (setq iterations _iterations))
  (:fit (X y)
        (let ((n_data (send X :get-val 'dim0))
              (dim (send X :get-val 'dim1))
              X_
              choice
              pred
              err
              dw)
          (setq X_ (make-matrix n_data (+ dim 1)))
          (dotimes (i dim) (setf (matrix-column X_ i) (matrix-column X i)))
          (setf (matrix-column X_ dim) (get-vector n_data 1))
          (setq w (coerce (get-vector (+ dim 1) 1) float-vector))
          (dotimes (i iterations)
            (setq choice (random n_data))
            ; (print (matrix-row X_ choice))
            (setq pred (v. (matrix-row X_ choice) w))
            (setq err (- (aref y choice 0) pred))
            ; (print (* eta err))
            (setq dw (scale (* eta err) (matrix-row X_ choice)))
            (v+ w dw)
            )
          ))
  ; (:predict)
  ; (:score)
  )


(defun main ()
  ; train data
  (setq xtr1 (load-data "../data/Train1.txt"))
  (setq ytr1 (get-vector (send xtr1 :get-val 'dim0) 0))
  (setq xtr2 (load-data "../data/Train2.txt"))
  (setq ytr2 (get-vector (send xtr2 :get-val 'dim0) 1))
  ; test data
  (setq xts1 (load-data "../data/Test1.txt"))
  (setq yts1 (get-vector (send xts1 :get-val 'dim0) 0))
  (setq xts2 (load-data "../data/Test2.txt"))
  (setq yts2 (get-vector (send xts2 :get-val 'dim0) 1))
  )


(main)
(setq *l* (instance lms :init 0.001 10000))
(send *l* :fit xtr1 ytr1)
(print (send *l* :get-val 'w))
