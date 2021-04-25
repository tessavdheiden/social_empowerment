import pickle
import sys, math
from copy import copy, deepcopy
import numpy as np
import gym
from gym.utils import colorize, seeding, EzPickle

NUM_TILES_FOR_AVG = 5  # The number of tiles before and after to takeinto account for angle
MIN_SEGMENT_LENGHT = 8
TILE_NAME = 'tile'

# # Warning, Not optimized for training
SHOW_BETA_PI_ANGLE = 0  # Shows the direction of the beta+pi/2 angle in each tile
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0             # Track scale
TRACK_RAD = 900/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 2000/SCALE  # Game over boundary
FPS = 50                # Frames per second
ZOOM = 2.7              # Camera zoom
ZOOM_FOLLOW = True      # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
SHOW_JOINTS               = 0       # Shows joints in white

class RoadCreator(gym.Env, EzPickle):

    def __init__(self):
        pass

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []

    def _to_relative(self, id):
        return id - (self.info['track'] < self.info[id]['track']).sum()

    def _get_track(self, num_checkpoints, track_rad=900 / SCALE, x_bias=0, y_bias=0):

        # num_checkpoints = 12

        # Create checkpoints
        checkpoints = []
        for c in range(num_checkpoints):
            alpha = 2 * math.pi * c / num_checkpoints + np.random.uniform(0, 2 * math.pi * 1 / num_checkpoints)
            rad = np.random.uniform(track_rad / 3, track_rad)
            if c == 0:
                alpha = 0
                rad = 1.5 * track_rad
            if c == num_checkpoints - 1:
                alpha = 2 * math.pi * c / num_checkpoints
                self.start_alpha = 2 * math.pi * (-0.5) / num_checkpoints
                rad = 1.5 * track_rad
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * track_rad, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi: beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi: beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3: beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4: break
            no_freeze -= 1
            if no_freeze == 0: break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose > 0:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        track = [[a, b, x + x_bias * 2, y + y_bias * 2] for a, b, x, y in track]
        track = [[track[i - 1], track[i]] for i in range(len(track))]
        return track

    def _get_possible_candidates_for_obstacles(self):
        return list(range(len(self.track)))

    def _create_info(self):
        '''
        Creates the matrix with the information about the track points,
        whether they are at the end of the track, if they are intersections
        '''
        # Get if point is at the end
        info = np.zeros((sum(len(t) for t in self.tracks)), dtype=[
            ('track', 'int'),
            ('end', 'bool'),
            ('begining', 'bool'),
            ('intersection', 'bool'),
            ('intersection_id', 'int'),
            ('t', 'bool'),
            ('x', 'bool'),
            ('start', 'bool'),
            ('used', 'bool'),
            ('angle', 'float16'),
            ('ang_class', 'float16'),
            ('lanes', np.ndarray),
            ('count_left', 'int'),
            ('count_right', 'int'),
            ('count_left_delay', 'int'),
            ('count_right_delay', 'int'),
            ('visited', bool),
            # ('obstacles',np.ndarray)])
            ('obstacles', bool)])

        info['ang_class'] = np.nan
        info['intersection_id'] = -1
        info['obstacles'] = False

        for i in range(len(info)):
            info[i]['lanes'] = [True, True]

        for i in range(1, len(self.tracks)):
            track = self.tracks[i]
            info[len(self.tracks[i - 1]):len(self.tracks[i - 1]) + len(track)][
                'track'] = i  # This wont work for num_tracks > 2
            for j in range(len(track)):
                pos = j + len(self.tracks[i - 1])
                p = track[j]
                next_p = track[(j + 1) % len(track)]
                last_p = track[j - 1]
                if np.array_equal(p[1], next_p[0]) == False:
                    # it is at the end
                    info[pos]['end'] = True
                elif np.array_equal(p[0], last_p[1]) == False:
                    # it is at the start
                    info[pos]['start'] = True

        # Trying to get all intersections
        intersections = set()
        if self.num_tracks > 1:
            for pos, point1 in enumerate(self.tracks[0][:, 1, 2:]):
                d = np.linalg.norm(self.track[len(self.tracks[0]):, 1, 2:] - point1, axis=1)
                if d.min() <= 2.05 * TRACK_WIDTH:
                    intersections.add(pos)

            intersections = list(intersections)
            intersections.sort()
            track_len = len(self.tracks[0])

            def backwards():
                me = intersections[-1]
                del intersections[-1]
                if len(intersections) == 0:
                    return [me]
                else:
                    if (me - 1) % track_len == intersections[-1]:
                        return [me] + backwards()
                    else:
                        return [me]

            def forward():
                me = intersections[0]
                del intersections[0]
                if len(intersections) == 0:
                    return [me]
                else:
                    if (me + 1) % track_len == intersections[0]:
                        return [me] + forward()
                    else:
                        return [me]

            groups = []
            tmp_lst = []
            while len(intersections) != 0:
                me = intersections[0]
                tmp_lst = tmp_lst + backwards()
                if len(intersections) != 0:
                    if (me - 1) % track_len == intersections[-1]:
                        tmp_lst = tmp_lst + forward()

                groups.append(tmp_lst)
                tmp_lst = []

            for group in groups:
                min_dist_idx = None
                min_dist = 1e10
                for idx in group:
                    d = np.linalg.norm(self.track[track_len:, 1, 2:] - self.track[idx, 1, 2:], axis=1)
                    if d.min() < min_dist:
                        min_dist = d.min()
                        min_dist_idx = idx

                if min_dist <= TRACK_WIDTH:
                    intersections.append(min_dist_idx)

            info['intersection'][list(intersections)] = True

            # Classifying intersections
            for idx in intersections:
                point = self.track[idx, 1, 2:]
                d = np.linalg.norm(self.track[:, 1, 2:] - point, axis=1)
                argmin = d[info['track'] != 0].argmin()
                filt = np.where(d < TRACK_WIDTH * 2.5)

                # TODO ignore intersections with angles of pi/2

                if info[filt]['start'].sum() - info[filt]['end'].sum() != 0:
                    info[idx]['t'] = True
                    info[argmin + track_len]['t'] = True
                else:
                    # the sum can be zero because second tracks are not cutted in case of x
                    info[idx]['x'] = True
                    info[argmin + track_len]['x'] = True

                    # Getting angles of curves
        max_idxs = []
        self.track[:, 0, 1] = np.mod(self.track[:, 0, 1], 2 * math.pi)
        self.track[:, 1, 1] = np.mod(self.track[:, 1, 1], 2 * math.pi)
        for num_track in range(self.num_tracks):

            track = self.tracks[num_track]
            angles = track[:, 0, 1] - track[:, 1, 1]
            inters = np.logical_or(info[info['track'] == num_track]['t'], info[info['track'] == num_track]['x'])

            track_len_compl = (info['track'] < num_track).sum()
            track_len = len(track)

            while np.abs(angles).max() != 0.0:
                max_rel_idx = np.abs(angles).argmax()

                rel_idxs = [(max_rel_idx + j) % track_len for j in range(-NUM_TILES_FOR_AVG, NUM_TILES_FOR_AVG)]
                idxs_safety = [(max_rel_idx + j) % track_len for j in
                               range(-NUM_TILES_FOR_AVG * 2, NUM_TILES_FOR_AVG * 2)]

                if (inters[idxs_safety] == True).sum() == 0:
                    max_idxs.append(max_rel_idx + track_len_compl)
                    angles[rel_idxs] = 0.0
                else:
                    angles[max_rel_idx] = 0.0

        info['angle'][max_idxs] = self.track[max_idxs, 0, 1] - self.track[max_idxs, 1, 1]

        ######### populating intersection_id
        intersection_dict = {}

        # Remove keys which are to close
        intersection_keys = np.where(info['intersection'])[0]
        intersection_vals = np.where((info['x']) | (info['t']))[0]

        for val in intersection_vals:
            tmp = self.track[intersection_keys][:, 1, 2:]
            elm = self.track[val, 1, 2:]
            d = np.linalg.norm(tmp - elm, axis=1)
            if d.min() > TRACK_WIDTH * 2:
                if self.verbose > 0:
                    print("the closest intersection is too far away")
            else:
                k = intersection_keys[d.argmin()]

                if k in intersection_dict.keys():
                    pass
                else:
                    intersection_dict[k] = []

                intersection_dict[k].append(val)

        self.intersection_dict = intersection_dict

        for k, v in self.intersection_dict.items():
            info['intersection_id'][[k] + v] = k
        del self.intersection_dict
        ##############################################

        self.info = info

    def _set_lanes(self):
        if self.num_lanes_changes > 0 and self.num_lanes > 1:
            rm_lane = 0  # 1 remove lane, 0 keep lane
            lane = 0  # Which lane will be removed
            changes = np.sort(self.np_random.randint(0, len(self.track), self.num_lanes_changes))

            # check in changes work
            # There must be no change at least 50 pos before and end and after a start
            changes_bad = []
            for pos, idx in enumerate(changes):
                start_from = sum(self.info['track'] < self.info[idx]['track'])
                until = sum(self.info['track'] == self.info[idx]['track'])
                changes_in_track = np.subtract(changes, start_from)
                changes_in_track = changes_in_track[(changes_in_track < until) * (changes_in_track > 0)]
                idx_relative = idx - start_from

                if sum(((changes_in_track - idx) > 0) * (
                        (changes_in_track - idx) < 10)) > 0:  # TODO wont work when at end of track
                    changes_bad.append(idx)
                    next

                track_info = self.info[self.info['track'] == self.info[idx]['track']]
                for i in range(50 + 1):
                    if track_info[(idx_relative + i) % len(track_info)]['end'] or track_info[idx_relative - i]['start']:
                        changes_bad.append(idx)
                        break

            if len(changes_bad) > 0:
                changes = np.setdiff1d(changes, changes_bad)

            counter = 0  # in order to avoid more than max number of single lanes tiles
            for i, point in enumerate(self.track):
                change = True if i in changes else False
                rm_lane = (rm_lane + change) % 2

                if change and rm_lane == 1:  # if it is time to change and the turn is to remove lane
                    lane = np.random.randint(0, 2, 1)[0]

                if rm_lane:
                    self.info[i]['lanes'][lane] = False
                    counter += 1
                else:
                    counter = 0

                # Change if end/inter of or if change prob
                if self.info[i]['end'] or self.info[i]['start'] or counter > self.max_single_lane:
                    rm_lane = 0

            # Avoiding any change of lanes in last and beginning part of a track
            for num_track in range(self.num_tracks):
                for lane in range(self.num_lanes):
                    for i in range(10):
                        i %= len(self.tracks[num_track])
                        self.info[self.info['track'] == num_track][+i]['lanes'][lane] = True
                        self.info[self.info['track'] == num_track][-i]['lanes'][lane] = True

    def _remove_unfinished_roads(self):
        n = 0
        to_remove = set()
        # The problem only appears in track1
        while n < len(self.tracks[0]):
            prev_tile = self.tracks[0][n - 2]
            tile = self.tracks[0][n - 1]
            next_tile = self.tracks[0][n]

            if any(tile[0] != prev_tile[1]) or any(tile[1] != next_tile[0]):
                to_remove.update(n)
                n -= 1
            else:
                n += 1
        self.tracks[0] = np.delete(self.tracks[0], list(to_remove), axis=0)

        if len(self.tracks[1]) < 5:
            self.tracks[1] = np.delete(self.tracks[1], range(len(self.tracks[1])), axis=1)

    def _choice_random_track_from_file(self):
        idx = np.random.choice(self.tracks_df.index)
        return idx

    def _generate_track(self):
        if self.load_tracks_from is not None:
            idx = self._choice_random_track_from_file()
            try:
                dic = pickle.load(open(self.load_tracks_from + '/' + str(idx) + ".pkl", 'rb'))
            except Exception as e:
                print("######## Error ########")
                print("error loading the track", str(idx))
                print(e)
                return False
            else:
                self.track = dic['track']
                self.tracks = dic['tracks']
                self.info = dic['info']
                self.obstacle_contacts = np.zeros((len(self.obstacles_poly)), dtype=
                [('count', int), ('count_delay', int), ('visited', bool)])

                self.info[[
                    'count_left',
                    'count_right',
                    'count_right_delay',
                    'count_left_delay']] = 0
                self.info['visited'] = False

                return True
        else:
            tracks = []
            cp = 12
            for _ in range(self.num_tracks):
                # The following variables allow for more complex tracks but, it is also
                # harder to controll their properties and correct behaviour
                track = self._get_track(int(cp * (1 ** _)))  # ,x_bias=-40*_,y_bias=40*_)
                if not track or len(track) == 0: return track
                track = np.array(track)
                if _ > 0 and False:
                    # adding rotation to decrease number of overlaps
                    theta = np.random.uniform() * 2 * np.pi
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    track[:, 0, 2:] = (R @ track[:, 0, 2:].T).T
                    track[:, 1, 2:] = (R @ track[:, 1, 2:].T).T
                    track[:, :2] += theta
                tracks.append(track)

            self.tracks = tracks
            if self.num_tracks > 1:
                if self._remove_roads() == False: return False
                self._remove_unfinished_roads()

            if self.tracks[0].size <= 5:
                return False
            if self.num_tracks > 1:
                if self.tracks[1].size <= 5:
                    return False
                if self.tracks[0].shape[1:] != self.tracks[1].shape[1:]:
                    return False

                self.track = np.concatenate(self.tracks)
            else:
                self.track = np.array(self.tracks[0])

            self._create_info()
            # Avoid lonely tiles at the begining of track
            if self.info[0]['intersection_id'] != -1: return False

            self._set_lanes()

    def _create_single_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + np.random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break

        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.box2d.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def _create_track(self):

        Ok = self._generate_track()
        if Ok is False:
            return False

        # Red-white border on hard turns
        borders = []
        if True:
            for track in self.tracks:
                border = [False] * len(track)
                for i in range(1, len(track)):
                    good = True
                    oneside = 0
                    for neg in range(BORDER_MIN_COUNT):
                        beta1 = track[i - neg][1][1]
                        beta2 = track[i - neg][0][1]
                        good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                        oneside += np.sign(beta1 - beta2)
                    good &= abs(oneside) == BORDER_MIN_COUNT
                    border[i] = good
                for i in range(len(track)):
                    for neg in range(BORDER_MIN_COUNT):
                        # TODO ERROR, sometimes list index out of range
                        border[i - neg] |= border[i]
                borders.append(border)

            # Creating borders for printing
            pos = 0
            for j in range(self.num_tracks):
                track = self.tracks[j]
                border = borders[j]
                for i in range(len(track)):
                    alpha1, beta1, x1, y1 = track[i][1]
                    alpha2, beta2, x2, y2 = track[i][0]
                    if border[i]:
                        side = np.sign(beta2 - beta1)

                        c = 1

                        # Addapting border to appear at the right widht when there are different number of lanes
                        if self.num_lanes > 1:
                            if side == -1 and self.info[pos]['lanes'][0] == False: c = 0
                            if side == +1 and self.info[pos]['lanes'][1] == False: c = 0

                        b1_l = (
                        x1 + side * TRACK_WIDTH * c * math.cos(beta1), y1 + side * TRACK_WIDTH * c * math.sin(beta1))
                        b1_r = (x1 + side * (TRACK_WIDTH * c + BORDER) * math.cos(beta1),
                                y1 + side * (TRACK_WIDTH * c + BORDER) * math.sin(beta1))
                        b2_l = (
                        x2 + side * TRACK_WIDTH * c * math.cos(beta2), y2 + side * TRACK_WIDTH * c * math.sin(beta2))
                        b2_r = (x2 + side * (TRACK_WIDTH * c + BORDER) * math.cos(beta2),
                                y2 + side * (TRACK_WIDTH * c + BORDER) * math.sin(beta2))
                        self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0), 0, 0))
                    pos += 1

        # Create tiles
        for j in range(len(self.track)):
            obstacle = np.random.binomial(1, 0)
            alpha1, beta1, x1, y1 = self.track[j][1]
            alpha2, beta2, x2, y2 = self.track[j][0]

            # drawing angles of old config, the
            # black line is the angle (NOT WORKING)
            if SHOW_BETA_PI_ANGLE:
                if self.track_lanes == None: self.track_lanes = []
                p1x = x1 + np.cos(beta1) * 0.2
                p1y = y1 + np.sin(beta1) * 0.2
                p2x = x1 + np.cos(beta1) * 0.2 + np.cos(beta1 + np.pi / 2) * 2
                p2y = y1 + np.sin(beta1) * 0.2 + np.sin(beta1 + np.pi / 2) * 2
                p3x = x1 - np.cos(beta1) * 0.2 + np.cos(beta1 + np.pi / 2) * 2
                p3y = y1 - np.sin(beta1) * 0.2 + np.sin(beta1 + np.pi / 2) * 2
                p4x = x1 - np.cos(beta1) * 0.2
                p4y = y1 - np.sin(beta1) * 0.2
                self.track_lanes.append([
                    [p1x, p1y],
                    [p2x, p2y],
                    [p3x, p3y],
                    [p4x, p4y]])

            for lane in range(self.num_lanes):
                if self.info[j]['lanes'][lane]:

                    joint = False  # to differentiate joints from normal tiles

                    r = 1 - ((lane + 1) % self.num_lanes)
                    l = 1 - ((lane + 2) % self.num_lanes)

                    # Get if it is the first or last
                    first = False  # first of lane
                    last = False  # last tile of line

                    if self.info[j]['end'] == False and self.info[j]['start'] == False:

                        # Getting if first tile of lane
                        # if last tile was from the same lane
                        info_track = self.info[self.info['track'] == self.info[j]['track']]
                        j_relative = j - sum(self.info['track'] < self.info[j]['track'])

                        if info_track[j_relative - 1]['track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[j_relative - 1]['lanes'][lane] == False:
                                first = True
                        if info_track[(j_relative + 1) % len(info_track)]['track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[(j_relative + 1) % len(info_track)]['lanes'][lane] == False:
                                last = True

                    road1_l = (x1 - (1 - last) * l * TRACK_WIDTH * math.cos(beta1),
                               y1 - (1 - last) * l * TRACK_WIDTH * math.sin(beta1))
                    road1_r = (x1 + (1 - last) * r * TRACK_WIDTH * math.cos(beta1),
                               y1 + (1 - last) * r * TRACK_WIDTH * math.sin(beta1))
                    road2_l = (x2 - (1 - first) * l * TRACK_WIDTH * math.cos(beta2),
                               y2 - (1 - first) * l * TRACK_WIDTH * math.sin(beta2))
                    road2_r = (x2 + (1 - first) * r * TRACK_WIDTH * math.cos(beta2),
                               y2 + (1 - first) * r * TRACK_WIDTH * math.sin(beta2))

                    vertices = [road1_l, road1_r, road2_r, road2_l]

                    if self.info[j]['end'] == True or self.info[j]['start'] == True:

                        points = []  # to store the new points
                        p3 = []  # in order to save all points 3 to create joints
                        for i in [0, 1]:  # because there are two point to do
                            # Get the closest point to a line make by the continuing trend of the original road points, the points will be the points
                            # under a radius r from line to avoid taking points far away in the other extreme of the track
                            # Remember the distance from a point p3 to a line p1,p2 is d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                            # p1=(x1,y1)+sin/cos, p2=(x2,y2)+sin/cos, p3=points in poly
                            if self.info[j]['end']:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r
                            else:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r

                            if len(p3) == 0:
                                max_idx = sum(sum(self.info[self.info['track'] == 0]['lanes'],
                                                  []))  # this will work because only seconday tracks have ends
                                p3_org = sum([x[0] for x in self.road_poly[:max_idx]], [])
                                # filter p3 by distance to p1 < TRACK_WIDTH*2
                                distance = TRACK_WIDTH * 2
                                not_too_close = \
                                np.where(np.linalg.norm(np.subtract(p3_org, p1), axis=1) >= TRACK_WIDTH / 3)[0]
                                while len(p3) == 0 and distance < PLAYFIELD:
                                    close = np.where(np.linalg.norm(np.subtract(p3_org, p1), axis=1) <= distance)[0]
                                    p3 = [p3_org[i] for i in np.intersect1d(close, not_too_close)]
                                    distance += TRACK_WIDTH

                            if len(p3) == 0:
                                raise RuntimeError('p3 lenght is zero')

                            d = (np.cross(np.subtract(p2, p1), np.subtract(p1, p3))) ** 2 / np.linalg.norm(
                                np.subtract(p2, p1))
                            points.append(p3[d.argmin()])

                        if self.info[j]['start']:
                            vertices = [points[0], points[1], road1_r, road1_l]
                        else:
                            vertices = [road2_r, road2_l, points[0], points[1]]
                        joint = True

                    test_set = set([tuple(p) for p in vertices])
                    if len(test_set) >= 3:
                        # TODO CHECK IF THIS AVOID THE ERROR OF ASSERTION COUNT >= 3
                        # TODO remove this try and find a way of really catching the errer
                        # try:
                        self.fd_tile.shape.vertices = vertices
                        t = self.box2d.CreateStaticBody(fixtures=self.fd_tile)
                        # except AssertionError as e:
                        # print(str(e))
                        # print(vertices)
                        # return False
                        t.userData = t
                        i = 0
                        # changing the following i for j achives different colors when visited tiles
                        c = 0.01 * (i % 3)
                        if joint and SHOW_JOINTS:
                            t.color = [1, 1, 1]
                        else:
                            # t.color = [ROAD_COLOR[0], ROAD_COLOR[1], ROAD_COLOR[2]]
                            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
                        t.road_visited = False
                        t.typename = TILE_NAME
                        t.road_friction = 1.0
                        t.id = j
                        t.lane = lane
                        t.fixtures[0].sensor = True
                        self.road_poly.append((vertices, t.color, t.id, t.lane))
                        self.road.append(t)
                    else:
                        print("saved from error")

        return True

    def _position_car_on_reset(self):
        """
        This function takes care of placing the car in a position
        at every reset call. This function should be modify to have
        the desired behaviour of where the car appears, do not
        re-spawn the car after the reset function has been called
        """
        self.place_agent(self.get_rnd_point_in_track())

    def _update_state(self, new_frame):
        if self.frames_per_state > 1:
            self.state[:, :, -1] = new_frame
            self.state = self.state[:, :, self._update_index]
        else:
            self.state = new_frame

    def _transform_action(self, action):
        if self.discretize_actions == "soft":
            raise NotImplementedError
        elif self.discretize_actions == "hard":
            # ("NOTHING", "LEFT", "RIGHT", "ACCELERATE", "BREAK")
            # angle, gas, break
            if action == 0: action = [0, 0, 0.0]  # Nothing
            if action == 1: action = [-1, 0, 0.0]  # Left
            if action == 2: action = [+1, 0, 0.0]  # Right
            if action == 3: action = [0, +1, 0.0]  # Accelerate
            if action == 4: action = [0, 0, 0.8]  # break

        return action

    def _remove_roads(self):

        if self.num_tracks > 1:
            def _get_section(first, last, track):
                sec = []
                pos = 0
                found = False
                while 1:
                    point = track[pos % track.shape[0], :, 2:]
                    if np.linalg.norm(point[1] - first) <= TRACK_WIDTH / 2:
                        found = True
                    if found:
                        sec.append(point)
                        if np.linalg.norm(point[1] - last) <= TRACK_WIDTH / 2:
                            break
                    pos = pos + 1
                    if pos / track.shape[0] >= 2: break
                if sec == []: return False
                return np.array(sec)

            THRESHOLD = TRACK_WIDTH * 2

            track1 = np.array(self.tracks[0])
            track2 = np.array(self.tracks[1])

            points1 = track1[:, :, [2, 3]]
            points2 = track2[:, :, [2, 3]]

            inter2 = np.array([x for x in points2 if
                               (np.linalg.norm(points1[:, 1, :] - x[1:], axis=1) <= TRACK_WIDTH / 3.5).sum() >= 1])

            intersections = []
            for i in range(inter2.shape[0]):
                if np.array_equal(inter2[i - 1, 1, :], inter2[i, 0, :]) == False or np.array_equal(inter2[i, 1, :],
                                                                                                   inter2[((
                                                                                                                   i + 1) % len(
                                                                                                           inter2)), 0,
                                                                                                   :]) == False:
                    intersections.append(inter2[i])
            intersections = np.array(intersections)

            # For each point in intersection
            # > get section of both roads
            # > For each point in section in second road
            # > > get min distance
            # > get max of distances
            # if max dist < threshold remove
            removed_idx = set()
            intersection_keys = []
            intersection_vals = []
            sec1_closer_to_center = None
            for i in range(intersections.shape[0]):
                _, first = intersections[i - 1]
                last, _ = intersections[i]

                sec1 = _get_section(first, last, track1)
                sec2 = _get_section(first, last, track2)

                sec1_distance_to_center = np.mean(np.linalg.norm(sec1[2:], axis=1))
                sec2_distance_to_center = np.mean(np.linalg.norm(sec2[2:], axis=1))

                if sec1 is not False and sec2 is not False:

                    remove = False
                    if sec1_distance_to_center > sec2_distance_to_center:
                        # sec1 is outside
                        if sec1_closer_to_center is False:
                            remove = True
                        else:
                            sec1_closer_to_center = False
                    else:
                        # sec1 is inside
                        if sec1_closer_to_center is True:
                            remove = True
                        else:
                            sec1_closer_to_center = True

                    if remove is False:
                        max_min_d = 0
                        remove = False
                        min_distances = []
                        for point in sec1[:, 1]:
                            dist = np.linalg.norm(sec2[:, 1] - point, axis=1).min()
                            min_distances.append(dist)
                            # min_d = dist if max_min_d < dist else max_min_d

                        min_distances = np.array(min_distances)

                        # if the max minimal distance is too small
                        if min_distances.max() < THRESHOLD * 2:
                            remove = True
                        # if the middle tiles of segment are too close to main track
                        elif len(min_distances) > 25 and (min_distances[10:-10].min() < TRACK_WIDTH * 3):
                            remove = True
                        # if the segment is smaller than MIN_SEGMENT_LENGHT
                        elif len(min_distances) < MIN_SEGMENT_LENGHT:
                            remove = True
                        # if there are more than 50 tiles very close to main track
                        elif len(min_distances) > 50 and (min_distances < TRACK_WIDTH * 2).sum() > 50:
                            remove = True

                    # Removing tiles
                    if remove:
                        for point in sec2:
                            idx = np.all(track2[:, :, [2, 3]] == point, axis=(1, 2))
                            removed_idx.update(np.where(idx)[0])
                    else:
                        key = np.where(
                            np.all(track1[:, :, [2, 3]] == sec1[0], axis=(1, 2)))[0]
                        val = np.where(
                            np.all(track2[:, :, [2, 3]] == sec2[0], axis=(1, 2)))[0] \
                              + len(track1)
                        intersection_keys.append(key[0])
                        intersection_vals.append(val[0])

                        key = np.where(
                            np.all(track1[:, :, [2, 3]] == sec1[-1], axis=(1, 2)))[0]
                        val = np.where(
                            np.all(track2[:, :, [2, 3]] == sec2[-1], axis=(1, 2)))[0] \
                              + len(track1)
                        intersection_keys.append(key[0])
                        intersection_vals.append(val[0])

            track2 = np.delete(track2, list(removed_idx), axis=0)  # efficient way to delete them from np.array

            self.intersections = intersections

            if len(track1) == 0 or len(track2) == 0:
                return False

            self.tracks[0] = track1
            self.tracks[1] = track2

            return True

