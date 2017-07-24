package com.viglink.merchantorpublisher

import java.io.File

import cc.mallet.classify.{Classification, Classifier, NaiveBayesTrainer}
import cc.mallet.pipe._
import cc.mallet.pipe.iterator.FileIterator
import cc.mallet.types.{Instance, InstanceList}

import scala.io.Source

class MerchantPublisherClassifier {

  var model: Option[Classifier] = None

  def loadModel(clf: Classifier) = {
    this.model = Some(clf)
  }

  def train(trainingDataDir: String, trainTestRatio: Array[Double]): Classifier = {

    val baseDir = new File(trainingDataDir)
    val fileList = baseDir.listFiles().filter(!_.getName.contains(".DS_Store"))

    val trainingDirs = fileList.map(f => new File(trainingDataDir + "/" + f.getName))

    val pipes = Array(
      new Target2Label(),
      new Input2CharSequence("UTF-8"),
      new CharSequence2TokenSequence(),
      new TokenSequenceLowercase(),
      new TokenSequenceRemoveStopwords(),
      new TokenSequence2FeatureSequence(),
      new FeatureSequence2FeatureVector()
      //new PrintInputAndTarget()
    )

    val instancePipe = new SerialPipes(pipes)

    //pass training examples through pipes
    val ilist = new InstanceList(instancePipe)
    ilist.addThruPipe(new FileIterator(trainingDirs, FileIterator.STARTING_DIRECTORIES))

    val ilists = ilist.split(trainTestRatio)

    val training = ilists(0)
    val testing = ilists(1)

    //val maxentTrainer = new MaxEntTrainer()
    //val classifier = maxentTrainer.train(training)

    val naiveBayesTrainer = new NaiveBayesTrainer()
    val classifier = naiveBayesTrainer.train(training)

    println("\n\nTRAINING RESULTS:")
    println(s"Training set size: ${training.size()}")
    println(s"Test set size: ${testing.size()}")

    println(s"Training accuracy: ${classifier.getAccuracy(training)}")
    println(s"Test accuracy: ${classifier.getAccuracy(testing)}")

    classifier
  }

  def predict(text: String, clf: Option[Classifier] = None): Classification = {

    //make sure there is some kind of model loaded
    clf match {
      case Some(c) => this.loadModel(c)
      case None => assert(this.model.isDefined)
    }

    val clfModel = this.model.get
    clfModel.classify(clfModel.getInstancePipe().instanceFrom(new Instance(text, "", "", "")))
  }

}

