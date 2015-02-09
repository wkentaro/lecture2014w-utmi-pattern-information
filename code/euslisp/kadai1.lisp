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
          ;; add bias to X
          (setq X_ (make-matrix n_data (+ dim 1)))
          (dotimes (i dim) (setf (matrix-column X_ i) (matrix-column X i)))
          (setf (matrix-column X_ dim) (get-vector n_data 1))
          ;; set and update weight
          (setq w (coerce (get-vector (+ dim 1) 1) float-vector))
          (dotimes (i iterations)
            (setq choice (random n_data))
            ; (print (matrix-row X_ choice))
            (setq pred (v. (matrix-row X_ choice) w))
            (setq err (- (aref y choice 0) pred))
            ; (print (* eta err))
            (setq dw (scale (* eta err) (matrix-row X_ choice)))
            (setq w (v+ w dw))
            )
          ))
  (:predict (X)
            (let ((n_data (send X :get-val 'dim0))
                  (dim (send X :get-val 'dim1))
                  X_
                  pred)
              ;; add bias to X
              (setq X_ (make-matrix n_data (+ dim 1)))
              (dotimes (i dim) (setf (matrix-column X_ i) (matrix-column X i)))
              (setf (matrix-column X_ dim) (get-vector n_data 1))
              (setq y_pred (get-vector n_data 0))
              (dotimes (i n_data)
                (setq pred (v. (matrix-row X_ i) w))
                (let ((a (* pred pred))
                      (b (* (- 1 pred) (- 1 pred))))
                  (setf (elt y_pred i) (if (< a b) 0 1)))
                )
              )
            y_pred)
  (:score (X y_true)
          (let ((y_pred (send self :predict X))
                (nac 0))
            (dotimes (i (length y_pred))
              (if (= (elt y_pred i) (elt y_true i)) (incf nac)))
            (setq score (/ (float nac) (length y_pred)))
            )
          score)
  )


(defun main ()
  ; train data
  (setq xtr1 (load-data "../../data/Train1.txt"))
  (setq xtr2 (load-data "../../data/Train2.txt"))
  (setq len1 (send xtr1 :get-val 'dim0))
  (setq len2 (send xtr2 :get-val 'dim0))
  (setq xtr (make-matrix (+ len1 len2) (send xtr1 :get-val 'dim1)))
  (dotimes (i len1) (setf (matrix-row xtr i) (matrix-row xtr1 i)))
  (dotimes (i len2) (setf (matrix-row xtr (+ i len1)) (matrix-row xtr2 i)))
  (setq ytr1 (get-vector (send xtr1 :get-val 'dim0) 0))
  (setq ytr2 (get-vector (send xtr2 :get-val 'dim0) 1))
  (setq ytr (get-vector (+ len1 len2) 0))
  (dotimes (i len1) (setf (elt ytr i) (elt ytr1 i)))
  (dotimes (i len2) (setf (elt ytr (+ i len1)) (elt ytr2 i)))
  ; test data
  (setq xts1 (load-data "../../data/Test1.txt"))
  (setq xts2 (load-data "../../data/Test2.txt"))
  (setq len1 (send xts1 :get-val 'dim0))
  (setq len2 (send xts2 :get-val 'dim0))
  (setq xts (make-matrix (+ len1 len2) (send xts1 :get-val 'dim1)))
  (dotimes (i len1) (setf (matrix-row xts i) (matrix-row xts1 i)))
  (dotimes (i len2) (setf (matrix-row xts (+ i len1)) (matrix-row xts2 i)))
  (setq yts1 (get-vector (send xts1 :get-val 'dim0) 0))
  (setq yts2 (get-vector (send xts2 :get-val 'dim0) 1))
  (setq yts (get-vector (+ len1 len2) 0))
  (dotimes (i len1) (setf (elt yts i) (elt yts1 i)))
  (dotimes (i len2) (setf (elt yts (+ i len1)) (elt yts2 i)))
  ;; train and predict
  (setq *lms* (instance lms :init 0.001 10000))
  (send *lms* :fit xtr ytr)
  ; (print (send *lms* :predict xts))
  ;; plot graph
  (setq w (send *lms* :get-val 'w))
  (setq a (* (/ 1. (elt w 1)) (- 0.5 (elt w 2))))
  (setq b (* (/ 1. (elt w 1)) (* -1 (elt w 0))))
  (setq gp-command-list (list "set xlabel 'x';"
                              "set ylabel 'y';"
                              "plot '../../data/Train1.txt' using 1:2;"
                              "replot '../../data/Train2.txt' using 1:2;"
                              "replot '../../data/Test1.txt' using 1:2;"
                              "replot '../../data/Test2.txt' using 1:2;"
                              (format nil "replot y=~Ax+~A;" a b)
                              "set term png;"
                              "set output 'kadai1.png';"
                              "replot;"))
  (unix::system (format nil "gnuplot -e \"~A\""
    (let ((str "")) (dolist (gpc gp-command-list) (setq str (format nil "~A ~A" str gpc))) str)))
  (format t "score: ~A~%" (send *lms* :score xts yts))
  )

(main)
(exit)

