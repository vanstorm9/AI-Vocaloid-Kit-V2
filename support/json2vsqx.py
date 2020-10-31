# -*- coding: utf-8 -*-
import xml.dom.minidom
import json
import sys

miku_go = {
        "あ": "a",    "い": "i",     "う": "M",     "え": "e",    "お": "o",
        "か": "k a",  "き": "k' i",  "く": "k M",   "け": "k e",  "こ": "k o",
        "さ": "s a",  "し": "S i",   "す": "s M",   "せ": "s e",  "そ": "s o",
        "た": "t a",  "ち": "tS i",  "つ": "ts M",  "て": "t e",  "と": "t o",
        "な": "n a",  "に": "J i",   "ぬ": "n M",   "ね": "n e",  "の": "n o",
        "は": "h a",  "ひ": "C i",   "ふ": "p\\ M", "へ": "h e",  "ほ": "h o",
        "ま": "m a",  "み": "m i",  "む": "m M",   "め": "m e",  "も": "m o",
        "ら": "4 a",  "り": "4' i",  "る": "4 M",   "れ": "4 e",  "ろ": "4 o",            
        "が": "g a",  "ぎ": "g' i",  "ぐ": "g M",   "げ": "g e",  "ご": "g o",
        "ざ": "dz a", "じ": "dZ i",  "ず": "dz M",  "ぜ": "dZ e", "ぞ": "dz o",
        "だ": "d a",  "ぢ": "dZ i",  "づ": "dz M",  "で": "d e",  "ど": "d o",
        "ば": "b a",  "び": "b' i",  "ぶ": "b M",   "べ": "b e",  "ぼ": "b o",
        "ぱ": "p a",  "ぴ": "p' i",  "ぷ": "p M",   "ぺ": "p e",  "ぽ": "p o",
        "や": "j a",  "ゆ": "j M",   "よ": "j o",
        "わ": "w a",  "ゐ": "w i",   "ゑ": "w e",   "を": "o",    "ん": "N\\", 
        "ふぁ": "p\ a", "つぁ": "ts a",
        "うぃ": "w i",  "すぃ": "s i",   "ずぃ": "dz i", "つぃ": "ts i",  "てぃ": "t' i",
        "でぃ": "d' i", "ふぃ": "p\' i",
        "とぅ": "t M",  "どぅ": "d M",
        "いぇ": "j e",  "うぇ": "w e",   "きぇ": "k' e", "しぇ": "S e",   "ちぇ": "tS e",
        "つぇ": "ts e", "てぇ": "t' e",  "にぇ": "J e",  "ひぇ": "C e",   "みぇ": "m' e",
        "りぇ": "4' e", "ぎぇ": "g' e",  "じぇ": "dZ e", "でぇ": "d' e",  "びぇ": "b' e",
        "ぴぇ": "p' e", "ふぇ": "p\ e",
        "うぉ": "w o",  "つぉ": "ts o",  "ふぉ": "p\ o",
        "きゃ": "k' a", "しゃ": "S a",   "ちゃ": "tS a", "てゃ": "t' a",  "にゃ": "J a",
        "ひゃ": "C a",  "みゃ": "m' a",  "りゃ": "4' a", "ぎゃ": "N' a",  "じゃ": "dZ a",
        "でゃ": "d' a", "びゃ": "b' a",  "ぴゃ": "p' a", "ふゃ": "p\' a",
        "きゅ": "k' M", "しゅ": "S M",   "ちゅ": "tS M", "てゅ": "t' M",  "にゅ": "J M",
        "ひゅ": "C M",  "みゅ": "m' M",  "りゅ": "4' M", "ぎゅ": "g' M",  "じゅ": "dZ M",
        "でゅ": "d' M", "びゅ": "b' M",  "ぴゅ": "p' M", "ふゅ": "p\' M",
        "きょ": "k' o", "しょ": "S o",   "ちょ": "tS o", "てょ": "t' o",  "にょ": "J o",
        "ひょ": "C o",  "みょ": "m' o",  "りょ": "4' o", "ぎょ": "N' o",  "じょ": "dZ o",
        "でょ": "d' o", "びょ": "b' o",  "ぴょ": "p' o"            
        }

def json2vsqx(data):
    resolution_value = str(data["resolution"])
    format_value = str(data["format"])
    tracks_value = str(data["tracks"])

    doc = xml.dom.minidom.Document()
    vsq3 = doc.createElementNS('http://www.yamaha.co.jp/vocaloid/schema/vsq3/', 'vsq3')
    vsq3.setAttribute("xmlns", "http://www.yamaha.co.jp/vocaloid/schema/vsq3/")
    vsq3.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    vsq3.setAttribute("xsi:schemaLocation", "http://www.yamaha.co.jp/vocaloid/schema/vsq3/ vsq3.xsd")

# vender
    vender = doc.createElement('vender')
    vender_value = doc.createCDATASection("Yamaha corporation")
    vender.appendChild(vender_value)
    vsq3.appendChild(vender)
# version
    version = doc.createElement('version')
    version_value = doc.createCDATASection("3.0.0.11")
    version.appendChild(version_value)
    vsq3.appendChild(version)
#vVoiceTable
    vVoiceTable = doc.createElement('vVoiceTable')
    vVoice = doc.createElement('vVoice')
    #vBS
    vBS = doc.createElement('vBS')
    vBS_text = doc.createTextNode(u'0')
    vBS.appendChild(vBS_text)
    vVoice.appendChild(vBS)
    #vPC
    vPC = doc.createElement('vPC')
    vPC_text = doc.createTextNode(u'0')
    vPC.appendChild(vPC_text)
    vVoice.appendChild(vPC)
    #compID
    compID = doc.createElement('compID')
    compID_value = doc.createCDATASection("HOGEHOGEHOGEHOGE")       # ここは変更が必要？
    compID.appendChild(compID_value)
    vVoice.appendChild(compID)
    #vVoiceName
    vVoiceName = doc.createElement('vVoiceName')
    vVoiceName_value = doc.createCDATASection("MIKU_V3_Original")
    vVoiceName.appendChild(vVoiceName_value)
    vVoice.appendChild(vVoiceName)
    #vVoiceParam
    vVoiceParam = doc.createElement('vVoiceParam')
    #bre
    bre = doc.createElement('bre')
    bre_text = doc.createTextNode(u'0')
    bre.appendChild(bre_text)
    vVoiceParam.appendChild(bre)
    #bri
    bri = doc.createElement('bri')
    bri_text = doc.createTextNode(u'0')
    bri.appendChild(bri_text)
    vVoiceParam.appendChild(bri)
    #cle
    cle = doc.createElement('cle')
    cle_text = doc.createTextNode(u'0')
    cle.appendChild(cle_text)
    vVoiceParam.appendChild(cle)
    #gen
    gen = doc.createElement('gen')
    gen_text = doc.createTextNode(u'0')
    gen.appendChild(gen_text)
    vVoiceParam.appendChild(gen)
    #ope
    ope = doc.createElement('ope')
    ope_text = doc.createTextNode(u'0')
    ope.appendChild(ope_text)
    vVoiceParam.appendChild(ope)
    vVoice.appendChild(vVoiceParam)
    vVoiceTable.appendChild(vVoice)
    vsq3.appendChild(vVoiceTable)

#mixer
    mixer = doc.createElement('mixer')
    #masterUnit
    masterUnit = doc.createElement('masterUnit')
    #outDev
    outDev = doc.createElement('outDev')
    outDev_text = doc.createTextNode(u'0')
    outDev.appendChild(outDev_text)
    masterUnit.appendChild(outDev)
    #retLevel
    retLevel = doc.createElement('retLevel')
    retLevel_text = doc.createTextNode(u'0')
    retLevel.appendChild(retLevel_text)
    masterUnit.appendChild(retLevel)
    #vol
    vol = doc.createElement('vol')
    vol_text = doc.createTextNode(u'0')
    vol.appendChild(vol_text)
    masterUnit.appendChild(vol)
    mixer.appendChild(masterUnit)

    #vsUnit
    vsUnit = doc.createElement('vsUnit')
    #vsTrackNo
    vsTrackNo = doc.createElement('vsTrackNo')
    vsTrackNo_text = doc.createTextNode(u'0')
    vsTrackNo.appendChild(vsTrackNo_text)
    vsUnit.appendChild(vsTrackNo)
    #inGain
    inGain = doc.createElement('inGain')
    inGain_text = doc.createTextNode(u'0')
    inGain.appendChild(inGain_text)
    vsUnit.appendChild(inGain)
    #sendLevel
    sendLevel = doc.createElement('sendLevel')
    sendLevel_text = doc.createTextNode(u'-898')
    sendLevel.appendChild(sendLevel_text)
    vsUnit.appendChild(sendLevel)
    #sendEnable
    sendEnable = doc.createElement('sendEnable')
    sendEnable_text = doc.createTextNode(u'0')
    sendEnable.appendChild(sendEnable_text)
    vsUnit.appendChild(sendEnable)
    #mute
    mute = doc.createElement('mute')
    mute_text = doc.createTextNode(u'0')
    mute.appendChild(mute_text)
    vsUnit.appendChild(mute)
    #solo
    solo = doc.createElement('solo')
    solo_text = doc.createTextNode(u'0')
    solo.appendChild(solo_text)
    vsUnit.appendChild(solo)
    #pan
    pan = doc.createElement('pan')
    pan_text = doc.createTextNode(u'64')
    pan.appendChild(pan_text)
    vsUnit.appendChild(pan)
    #vol
    vol = doc.createElement('vol')
    vol_text = doc.createTextNode(u'0')
    vol.appendChild(vol_text)
    vsUnit.appendChild(vol)
    mixer.appendChild(vsUnit)

    #seUnit
    seUnit = doc.createElement('seUnit')
    #inGain
    inGain = doc.createElement('inGain')
    inGain_text = doc.createTextNode(u'0')
    inGain.appendChild(inGain_text)
    seUnit.appendChild(inGain)
    #sendLevel
    sendLevel = doc.createElement('sendLevel')
    sendLevel_text = doc.createTextNode(u'-898')
    sendLevel.appendChild(sendLevel_text)
    seUnit.appendChild(sendLevel)
    #sendEnable
    sendEnable = doc.createElement('sendEnable')
    sendEnable_text = doc.createTextNode(u'0')
    sendEnable.appendChild(sendEnable_text)
    seUnit.appendChild(sendEnable)
    #mute
    mute = doc.createElement('mute')
    mute_text = doc.createTextNode(u'0')
    mute.appendChild(mute_text)
    seUnit.appendChild(mute)
    #solo
    solo = doc.createElement('solo')
    solo_text = doc.createTextNode(u'0')
    solo.appendChild(solo_text)
    seUnit.appendChild(solo)
    #pan
    pan = doc.createElement('pan')
    pan_text = doc.createTextNode(u'64')
    pan.appendChild(pan_text)
    seUnit.appendChild(pan)
    #vol
    vol = doc.createElement('vol')
    vol_text = doc.createTextNode(u'0')
    vol.appendChild(vol_text)
    seUnit.appendChild(vol)
    mixer.appendChild(seUnit)

    #karaokeUnit
    karaokeUnit = doc.createElement('karaokeUnit')
    #inGain
    inGain = doc.createElement('inGain')
    inGain_text = doc.createTextNode(u'0')
    inGain.appendChild(inGain_text)
    karaokeUnit.appendChild(inGain)
    #mute
    mute = doc.createElement('mute')
    mute_text = doc.createTextNode(u'0')
    mute.appendChild(mute_text)
    karaokeUnit.appendChild(mute)
    #solo
    solo = doc.createElement('solo')
    solo_text = doc.createTextNode(u'0')
    solo.appendChild(solo_text)
    karaokeUnit.appendChild(solo)
    #vol
    vol = doc.createElement('vol')
    vol_text = doc.createTextNode(u'-129')
    vol.appendChild(vol_text)
    karaokeUnit.appendChild(vol)
    mixer.appendChild(karaokeUnit)
    vsq3.appendChild(mixer)

#masterTrack        本当はテンポが細かく決まっている場合があるけど、今回はデフォルトにする
    masterTrack = doc.createElement('masterTrack')
    #seqName
    seqName = doc.createElement('seqName')
    seqName_text = doc.createCDATASection(u'none')
    seqName.appendChild(seqName_text)
    masterTrack.appendChild(seqName)
    #comment
    comment = doc.createElement('comment')
    comment_text = doc.createCDATASection(u'none')
    comment.appendChild(comment_text)
    masterTrack.appendChild(comment)
    #resolution
    resolution = doc.createElement('resolution')
    resolution_text = doc.createTextNode(str(resolution_value))
    resolution.appendChild(resolution_text)
    masterTrack.appendChild(resolution)
    #preMeasure
    preMeasure = doc.createElement('preMeasure')
    preMeasure_text = doc.createTextNode('4')
    preMeasure.appendChild(preMeasure_text)
    masterTrack.appendChild(preMeasure)
    #timeSig
    timeSig = doc.createElement('timeSig')
    #posMes
    posMes = doc.createElement('posMes')
    posMes_text = doc.createTextNode(u'0')
    posMes.appendChild(posMes_text)
    timeSig.appendChild(posMes)
    #nume
    nume = doc.createElement('nume')
    nume_text = doc.createTextNode(u'4')
    nume.appendChild(nume_text)
    timeSig.appendChild(nume)
    #denomi
    denomi = doc.createElement('denomi')
    denomi_text = doc.createTextNode(u'4')
    denomi.appendChild(denomi_text)
    timeSig.appendChild(denomi)
    masterTrack.appendChild(timeSig)
    #tempo
    tempo = doc.createElement('tempo')
    #posTick
    posTick = doc.createElement('posTick')
    posTick_text = doc.createTextNode(u'0')
    posTick.appendChild(posTick_text)
    tempo.appendChild(posTick)
    #bpm
    bpm = doc.createElement('bpm')
    bpm_text = doc.createTextNode(u'12000')
    bpm.appendChild(bpm_text)
    tempo.appendChild(bpm)
    masterTrack.appendChild(tempo)
    vsq3.appendChild(masterTrack)

#vsTrack
    vsTrack = doc.createElement('vsTrack')
    #vsTrackNo
    vsTrackNo = doc.createElement('vsTrackNo')
    vsTrackNo_text = doc.createTextNode(u'0')
    vsTrackNo.appendChild(vsTrackNo_text)
    vsTrack.appendChild(vsTrackNo)
    #trackName
    trackName = doc.createElement('trackName')
    trackName_text = doc.createCDATASection(u'Track')
    trackName.appendChild(trackName_text)
    vsTrack.appendChild(trackName)
    #comment
    comment = doc.createElement('comment')
    comment_text = doc.createCDATASection(u'Track')
    comment.appendChild(comment_text)
    vsTrack.appendChild(comment)
    #musicalPart
    musicalPart = doc.createElement('musicalPart')
    #posTick
    posTick = doc.createElement('posTick')
    posTick_text = doc.createTextNode(u'7680')
    posTick.appendChild(posTick_text)
    musicalPart.appendChild(posTick)
    #playTime
    playTime = doc.createElement('playTime')
    playTime_text = doc.createTextNode(u'614400')        # ここは伸ばす?
    playTime.appendChild(playTime_text)
    musicalPart.appendChild(playTime)
    #partName
    partName = doc.createElement('partName')
    partName_text = doc.createCDATASection(u'NewPart')
    partName.appendChild(partName_text)
    musicalPart.appendChild(partName)
    #comment
    comment = doc.createElement('comment')
    comment_text = doc.createCDATASection(u'New Musical Part')
    comment.appendChild(comment_text)
    musicalPart.appendChild(comment)

    #stylePlugin
    stylePlugin = doc.createElement('stylePlugin')
    #stylePluginID
    stylePluginID = doc.createElement('stylePluginID')
    stylePluginID_text = doc.createCDATASection(u'ACA9C502-A04B-42b5-B2EB-5CEA36D16FCE')
    stylePluginID.appendChild(stylePluginID_text)
    stylePlugin.appendChild(stylePluginID)
    #stylePluginName
    stylePluginName = doc.createElement('stylePluginName')
    stylePluginName_text = doc.createCDATASection(u'VOCALOID2 Compatible Style')
    stylePluginName.appendChild(stylePluginName_text)
    stylePlugin.appendChild(stylePluginName)
    #version
    version = doc.createElement('version')
    version_text = doc.createCDATASection(u'3.0.0.1')
    version.appendChild(version_text)
    stylePlugin.appendChild(version)
    musicalPart.appendChild(stylePlugin)

    #partStyle
    partStyle = doc.createElement('partStyle')
    #attr accent
    attr = doc.createElement('attr')
    attr.setAttribute("id", "accent")
    attr_text = doc.createTextNode(u'50')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr bendDep
    attr = doc.createElement('attr')
    attr.setAttribute("id", "bendDep")
    attr_text = doc.createTextNode(u'8')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr bendLen
    attr = doc.createElement('attr')
    attr.setAttribute("id", "bendLen")
    attr_text = doc.createTextNode(u'0')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr decay
    attr = doc.createElement('attr')
    attr.setAttribute("id", "decay")
    attr_text = doc.createTextNode(u'50')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr fallPort
    attr = doc.createElement('attr')
    attr.setAttribute("id", "fallPort")
    attr_text = doc.createTextNode(u'0')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr opening
    attr = doc.createElement('attr')
    attr.setAttribute("id", "opening")
    attr_text = doc.createTextNode(u'127')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    #attr risePort
    attr = doc.createElement('attr')
    attr.setAttribute("id", "risePort")
    attr_text = doc.createTextNode(u'0')
    attr.appendChild(attr_text)
    partStyle.appendChild(attr)
    musicalPart.appendChild(partStyle)

    #singer
    singer = doc.createElement('singer')
    #posTick
    posTick = doc.createElement('posTick')
    posTick_text = doc.createTextNode(u'0')
    posTick.appendChild(posTick_text)
    singer.appendChild(posTick)
    #vBS
    vBS = doc.createElement('vBS')
    vBS_text = doc.createTextNode(u'0')
    vBS.appendChild(vBS_text)
    singer.appendChild(vBS)
    #vPC
    vPC = doc.createElement('vPC')
    vPC_text = doc.createTextNode(u'0')
    vPC.appendChild(vPC_text)
    singer.appendChild(vPC)
    musicalPart.appendChild(singer)

#==========ここから楽譜============================
    POSTICK = 0
    DURTICK = 0
    for note in data["stream"]:
        if note["sub_type"] == "noteOn":
            VELOCITY = note["velocity"]
            A = note["tick"]
        if note["sub_type"] == "noteOff":
            if POSTICK == 0:
                POSTICK = A
            else:
                POSTICK = POSTICK + (A + DURTICK)
            NOTENUM = note["note_num"]
            DURTICK = note["tick"]
            LYRICS = note.get("lyrics", "ら")
            #print POSTICK, DURTICK, NOTENUM, VELOCITY
            #note
            note = doc.createElement('note')
            #posTick
            posTick = doc.createElement('posTick')
            posTick_text = doc.createTextNode(str(POSTICK))
            posTick.appendChild(posTick_text)
            note.appendChild(posTick)
            #durTick
            durTick = doc.createElement('durTick')
            durTick_text = doc.createTextNode(str(DURTICK))
            durTick.appendChild(durTick_text)
            note.appendChild(durTick)
            #noteNum
            noteNum = doc.createElement('noteNum')
            noteNum_text = doc.createTextNode(str(NOTENUM))
            noteNum.appendChild(noteNum_text)
            note.appendChild(noteNum)
            #velocity
            velocity = doc.createElement('velocity')
            velocity_text = doc.createTextNode(str(VELOCITY))
            velocity.appendChild(velocity_text)
            note.appendChild(velocity)
#======================歌詞
            #lyric
            lyric = doc.createElement('lyric')
            lyric_text = doc.createCDATASection(LYRICS)
            lyric.appendChild(lyric_text)
            note.appendChild(lyric)
            #phnms
            phnms = doc.createElement('phnms')
            phnms_text = doc.createCDATASection(miku_go.get(LYRICS, '4 a'))
            phnms.appendChild(phnms_text)
            note.appendChild(phnms)
#======================
            #noteStyle
            noteStyle = doc.createElement('noteStyle')
            #attr accent
            attr = doc.createElement('attr')
            attr.setAttribute("id", "accent")
            attr_text = doc.createTextNode(u'50')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr bendDep
            attr = doc.createElement('attr')
            attr.setAttribute("id", "bendDep")
            attr_text = doc.createTextNode(u'8')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr bendLen
            attr = doc.createElement('attr')
            attr.setAttribute("id", "bendLen")
            attr_text = doc.createTextNode(u'0')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr decay
            attr = doc.createElement('attr')
            attr.setAttribute("id", "decay")
            attr_text = doc.createTextNode(u'50')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr fallPort
            attr = doc.createElement('attr')
            attr.setAttribute("id", "fallPort")
            attr_text = doc.createTextNode(u'0')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr opening
            attr = doc.createElement('attr')
            attr.setAttribute("id", "opening")
            attr_text = doc.createTextNode(u'127')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr risePort
            attr = doc.createElement('attr')
            attr.setAttribute("id", "risePort")
            attr_text = doc.createTextNode(u'0')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr vibLen
            attr = doc.createElement('attr')
            attr.setAttribute("id", "vibLen")
            attr_text = doc.createTextNode(u'0')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            #attr vibType
            attr = doc.createElement('attr')
            attr.setAttribute("id", "vibType")
            attr_text = doc.createTextNode(u'0')
            attr.appendChild(attr_text)
            noteStyle.appendChild(attr)
            note.appendChild(noteStyle)
            musicalPart.appendChild(note)
#==================================================
    vsTrack.appendChild(musicalPart)
    vsq3.appendChild(vsTrack)

#seTrack
    seTrack = doc.createElement('seTrack')
    seTrack_text = doc.createTextNode(u'')
    seTrack.appendChild(seTrack_text)
    vsq3.appendChild(seTrack)

#karaokeTrack
    karaokeTrack = doc.createElement('karaokeTrack')
    karaokeTrack_text = doc.createTextNode(u'')
    karaokeTrack.appendChild(karaokeTrack_text)
    vsq3.appendChild(karaokeTrack)

#aux
    aux = doc.createElement('aux')
    # auxID 
    auxID = doc.createElement('auxID')
    auxID_text = doc.createCDATASection(u'AUX_VST_HOST_CHUNK_INFO')
    auxID.appendChild(auxID_text)
    aux.appendChild(auxID)
    # content 
    content = doc.createElement('content')
    content_text = doc.createCDATASection(u'VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=')
    content.appendChild(content_text)
    aux.appendChild(content)
    vsq3.appendChild(aux)
    doc.appendChild(vsq3)

    return doc
