const chest = require('./exercises/chest')
const back = require('./exercises/back')
const shoulder = require('./exercises/shoulder')
const arms = require('./exercises/arms')
const legs = require('./exercises/legs')
const core = require('./exercises/core')
const cardio = require('./exercises/cardio')

const exerciseData = { chest, back, shoulder, arms, legs, core, cardio }
const groupList = [chest, back, shoulder, arms, legs, core, cardio]

module.exports = { exerciseData, groupList }
