package com.viglink.merchantorpublisher

import java.io.File

import cc.mallet.types.Instance

import scala.io.Source


object Demo extends App {

  //get the absolute path of the training data folder in resources
  val trainingDataPath = new File("src/main/resources/trainingData/").getAbsolutePath

  //train model - not if test proportion is 0.0 then there will be no evaluation data shown
  val trainTestSplit = Array(100.0, 0.0)
  val clf = Classifier.train(trainingDataPath, trainTestSplit)

  //evaluate on a couple hold-out examples. Barney's is a merchant and avsforum is a publisher
  val evaluationDataPath = new File("src/main/resources/evaluationData/").getAbsolutePath
  val barneys_html = Source.fromFile(s"$evaluationDataPath/barneys_1.html").getLines().toList.mkString(" ")
  val avsforum_html = Source.fromFile(s"$evaluationDataPath/avsforum_1.html").getLines().toList.mkString(" ")

  //print classifier predictions
  val barneysPred = clf.classify(clf.getInstancePipe().instanceFrom(new Instance(barneys_html, "", "", "")))
  println("\n\nBARNEYS PAGE PREDICTION RESULTS:")
  println(barneysPred.getLabeling().getBestLabel().toString())

  println("\n\nAVSFORUM.COM PAGE PREDICTION RESULTS:")
  val petguidePred = clf.classify(clf.getInstancePipe().instanceFrom(new Instance(avsforum_html, "", "", "")))
  println(petguidePred.getLabeling().getBestLabel().toString())
}
