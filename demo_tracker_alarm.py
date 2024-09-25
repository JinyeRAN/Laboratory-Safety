import pandas as pd
import time
import requests
import json
import os
import cv2

from tools import CurrentTime


class Alarm(object):
    def __init__(self):
        datax = ('PeopleID', 'white_coat', 'glasses', 'times', 'all_times')
        self.Table = pd.DataFrame(columns=datax)
        self.SavePicturePath = '/workspace/data/' 
        self.DecetectionPeriods = 1 
        # self.DecetectionPeriods_F = 10
        self.MaxMissNum = self.DecetectionPeriods * 2
        # self.names = ['white_coat', 'glasses', 'gloves']
        self.names = ['white_coat', 'glasses']

        self.sign = True
        self.time_alarm = 0


    def AlramDetemine(self):
        res = self.Table[(self.Table['times'] > self.DecetectionPeriods)]
        # res = res[(res['white_coat'] < 1) | (res['glasses'] < 1) | (res['gloves'] < 1)]
        res = res[(res['white_coat'] < 1) | (res['glasses'] < 1)]
        return res

    def AlarmRestate(self):
        self.Table = self.Table.drop(self.Table[self.Table['all_times'] > self.MaxMissNum].index)
        # self.Table.loc[self.Table.times > self.DecetectionPeriods, 'white_coat':'times'] = 0

    def TimesAndAllRenew(self, PeopleID):
        indd = []
        for ID in PeopleID:
            ind = self.Table[self.Table.PeopleID == ID].index.tolist()
            self.Table.loc[ind, 'times'] = self.Table.loc[ind, 'times'] + 1
            self.Table.loc[ind, 'all_times'] = 0
            indd.append(ind[0])
        for other in self.Table.index:
            if not other in indd:
                self.Table.loc[other, 'all_times'] = self.Table.loc[other, 'all_times'] + 1

    def Vote(self, PeopleID, result, name):
        for ID, res in zip(PeopleID, result):
            if ID in self.Table['PeopleID'].values:
                ind = self.Table[self.Table.PeopleID == ID].index.tolist()[0]
                self.Table.loc[ind, name] = self.Table.loc[ind, name] or res
            else:
                # self.Table = self.Table.append(
                #     [{'PeopleID': ID, 'white_coat': 0, 'glasses': 0, 'gloves': 0, 'times': 0, 'all_times': 0}],
                #     ignore_index=True)
                self.Table = self.Table.append(
                    [{'PeopleID': ID, 'white_coat': 0, 'glasses': 0, 'times': 0, 'all_times': 0}],
                    ignore_index=True)
                ind = self.Table[self.Table.PeopleID == ID].index.tolist()[0]
                self.Table.loc[ind, name] = self.Table.loc[ind, name] or res

    def judge(self, track_result, det):
        result = []
        count = 0
        for people in track_result:
            people_bbox = people[:4]
            for catlogy in det:
                if self.compute_IOU(people_bbox, catlogy) > 0:
                    result.append(1)
                    break
            if len(result) == count:
                count += 1
                result.append(0)
                continue
            count += 1
        return result

    def compute_IOU(self, rec1, rec2):
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)

    def add_warning_data(self, caID,rtmp, AlarmCatalogy, PicturePath):
        params = {
            "equipment_id": rtmp,
            "warning_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "warning_type_code": AlarmCatalogy,
            "picture_path": PicturePath,
            'lab':caID
        }
        params = {"data": json.dumps(params)}
        url = 'http://10.3.50.250:8080/labor/api/apiAction!addWarningInfo'
        try:
            resp = requests.post(url, data=params)
            if 200 == resp.status_code:
                print("succuss data to database")
                return resp.text
            else:
                return resp.text
        except requests.ConnectionError:
            print("fail")
            return None

    def alarm(self, track, predict, Image0, id,rtmp):
        # print('label_time:',self.time_alarm)
        if self.sign:
            self.time_alarm = (int(str(CurrentTime())[0:2]) + 1)*10000
            self.sign = False
        else:
            pass
        peopleId = track[:, 4]
        for other, name in zip(predict, self.names):
            detection_result = self.judge(track, other)
            self.Vote(peopleId, detection_result, name)
        self.TimesAndAllRenew(peopleId)
        self.AlarmRestate()
        print(self.Table, flush=True)

        time_local = CurrentTime()
        # print('current time: ',time_local)
        if time_local > self.time_alarm:
            self.sign = True
            res = self.AlramDetemine()
            if res.empty:
                pass
            else:
                saveID = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                strx = 'no_'
                # for i, nameT in zip([res['white_coat'], res['glasses'], res['gloves']], self.names):
                for i, nameT in zip([res['white_coat'], res['glasses']], self.names):
                    for ii in i:
                        if 0 == ii:
                        # if 1 == ii:
                            strx = strx + nameT + '_'
                            break
                if not os.path.exists(os.path.join(self.SavePicturePath, str(id))):
                    os.mkdir(os.path.join(self.SavePicturePath, str(id)))
                SavePath = os.path.join(self.SavePicturePath, str(id), str(saveID) + '_' + strx + '.png')
                SendPath = os.path.join(str(id), str(saveID) + '_' + strx + '.png')
                # 添加结果
                cv2.imwrite(SavePath, Image0)
                # for alarm, name in zip(["PPE_1", "PPE_2", "PPE_3"], ['white_coat', 'glasses', 'gloves']):
                for alarm, name in zip(["PPE_1", "PPE_2"], ['white_coat', 'glasses']):
                    if name in strx:
                        self.add_warning_data(id,rtmp, str(name), SendPath)