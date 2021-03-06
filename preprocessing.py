import cv2 
import numpy as np
import os
from correction import process_images

def line_array(array):
    list_x_upper = []
    list_x_lower = []
    for y in range(5, len(array)-5):
        s_a, s_p = strtline(y, array)
        e_a, e_p = endline(y, array)
        if s_a>=7 and s_p>=5:
            list_x_upper.append(y)
        if e_a>=5 and e_p>=7:
            list_x_lower.append(y)
    return list_x_upper, list_x_lower

def line_array(array):
	list_x_upper = []
	list_x_lower = []
	for y in range(5, len(array)-5):
		s_a, s_p = strtline(y, array)
		e_a, e_p = endline(y, array)
		if s_a>=7 and s_p>=5:
			list_x_upper.append(y)
		if e_a>=5 and e_p>=7:
			list_x_lower.append(y)
	return list_x_upper, list_x_lower

def strtline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i > 3:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i==0:
			count_prev += 1  
	return count_ahead, count_prev

def endline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i==0:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i >3:
			count_prev += 1  
	return count_ahead, count_prev

def endline_word(y, array, a):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+2*a]:
		if i < 2:
			count_ahead+= 1  
	for i in array[y-a:y]:
		if i > 2:
			count_prev += 1  
	return count_prev ,count_ahead

def end_line_array(array, a):
	list_endlines = []
	for y in range(len(array)):
		e_p, e_a = endline_word(y, array, a)
		if e_a >= int(1.5*a) and e_p >= int(0.7*a):
			list_endlines.append(y)
	return list_endlines

def refine_endword(array):
	refine_list = []
	for y in range(len(array)-1):
		if array[y]+1 < array[y+1]:
			refine_list.append(array[y])
	return refine_list



def refine_array(array_upper, array_lower):
	upperlines = []
	lowerlines = []
	for y in range(len(array_upper)-1):
		if array_upper[y] + 5 < array_upper[y+1]:
			upperlines.append(array_upper[y]-10)
	for y in range(len(array_lower)-1):
		if array_lower[y] + 5 < array_lower[y+1]:
			lowerlines.append(array_lower[y]+10)

	upperlines.append(array_upper[-1]-10)
	lowerlines.append(array_lower[-1]+10)
	
	return upperlines, lowerlines

def letter_width(contours):
	letter_width_sum = 0
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > 20:
			x,y,w,h = cv2.boundingRect(cnt)
			letter_width_sum += w
			count += 1

	return letter_width_sum/count


def end_wrd_dtct(lines, i, bin_img, mean_lttr_width,final_thr,width):
	count_y = np.zeros(shape = width)
	for x in range(width):
		for y in range(lines[i][0],lines[i][1]):
			if bin_img[y][x] == 255:
				count_y[x] += 1
	end_lines = end_line_array(count_y, int(mean_lttr_width))
	endlines = refine_endword(end_lines)
	for x in endlines:
		final_thr[lines[i][0]:lines[i][1], x] = 255
	return endlines

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print ("Folder Exists")
    

def letter_seg(lines_img, x_lines, i,total_lines):
    copy_img = lines_img[i].copy()
    x_linescopy = x_lines[i].copy()
    
    letter_img = []
    letter_k = []
    
    contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x,y,w,h = cv2.boundingRect(cnt)
            letter_k.append((x,y,w,h))

    letter = sorted(letter_k, key=lambda student: student[0])
    
    word = 1
    letter_index = 0
    create_folder('./Data/Line'+str(i+1)+'/'+'Word'+str(word))
    total_lines.append((i+1,word))
    
    for e in range(len(letter)):
        if(letter[e][0]<x_linescopy[0]):
            letter_index += 1
            letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
            letter_img=letter_img_tmp
            cv2.imwrite('./Data/Line'+str(i+1)+'/'+'Word'+str(word)+'/'+str(letter_index)+'.jpg', letter_img)
           
        else:
            x_linescopy.pop(0)
            word += 1
            total_lines.append((i+1,word))
            create_folder('./Data/Line'+str(i+1)+'/Word'+str(word))
            letter_index = 1
            letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
            letter_img=letter_img_tmp
            cv2.imwrite('./Data/Line'+str(i+1)+'/'+'Word'+str(word)+'/'+str(letter_index)+'.jpg', letter_img)

def Preprocessing(src_img):
    total_lines=[]
    src_img = cv2.cv2.medianBlur(src_img,3)
    src_img = cv2.GaussianBlur(src_img,(3,3),0)
    cv2.imwrite('img.jpg',src_img)
    copy = src_img.copy()
    height = src_img.shape[0]
    width = src_img.shape[1]

    src_img = cv2.resize(copy, dsize =(720, int(720*height/width)), interpolation = cv2.INTER_AREA)

    height = src_img.shape[0]
    width = src_img.shape[1]

    bin_img = cv2.adaptiveThreshold(src_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,20)
    bin_img=cv2.bitwise_not(bin_img)
    bin_img2 = bin_img.copy()
    print('Preprocessing done')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contr_retrival = final_thr.copy()

    count_x = np.zeros(shape= (height))
    for y in range(height):
        for x in range(width):
            if bin_img[y][x] == 255 :
                count_x[y] = count_x[y]+1
    upper_lines, lower_lines = line_array(count_x)

    upperlines, lowerlines = refine_array(upper_lines, lower_lines)

    if len(upperlines)==len(lowerlines):
	    lines = []
	    for y in upperlines:
		    final_thr[y][:] = 255	
	    for y in lowerlines:
		    final_thr[y][:] = 255
	    for y in range(len(upperlines)):
		    lines.append((upperlines[y], lowerlines[y]))
	
    else:
	    return "Too much noise in image, unable to process.\nPlease try with another image."

    lines = np.array(lines)

    no_of_lines = len(lines)


    for i in range(no_of_lines):
        create_folder('Data/Line'+str(i+1))
            
    lines_img = []

    for i in range(no_of_lines):
        lines_img.append(bin_img2[lines[i][0]:lines[i][1], :])

    contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    mean_lttr_width = letter_width(contours)

    x_lines = []

    for i in range(len(lines_img)):
        x_lines.append(end_wrd_dtct(lines, i, bin_img, mean_lttr_width,final_thr,width))

    for i in range(len(x_lines)):
        x_lines[i].append(width)

    for i in range(len(lines)):
        letter_seg(lines_img, x_lines, i,total_lines)
    
    
    
    return total_lines




