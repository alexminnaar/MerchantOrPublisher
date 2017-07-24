package com.viglink.merchantorpublisher

import java.io.File

import cc.mallet.types.Instance

import scala.io.Source


object Demo extends App {

  //get the absolute path of the training data folder in resources
  val resourcesDirectory = new File("src/main/resources/TrainingData/")
  val trainingDataPath = resourcesDirectory.getAbsolutePath

  val trainTestSplit = Array(100.0,0.0)
  val clf = Classifier.train(trainingDataPath, trainTestSplit)

  val barneys_html = Source.fromFile("/users/alexminnaar/merchantOrPublisher_evaluation/barneys_1.html").getLines().toList.mkString(" ")
  val avsforum_html = Source.fromFile("/users/alexminnaar/merchantOrPublisher_evaluation/avsforum_1.html").getLines().toList.mkString(" ")

  val barneysPred = clf.classify(clf.getInstancePipe().instanceFrom(new Instance(barneys_html, "", "", "")))
  println("\n\nBARNEYS PAGE PREDICTION RESULTS:")
  println(barneysPred.getLabeling().getBestLabel().toString())

  println("\n\nAVSFORUM.COM PAGE PREDICTION RESULTS:")
  val petguidePred = clf.classify(clf.getInstancePipe().instanceFrom(new Instance(avsforum_html, "", "", "")))
  println(petguidePred.getLabeling().getBestLabel().toString())
}
