import pygame



def createJOI():
        # Path to the input videos
        video_path = 'C:/Users/Acer/yolov4/video/video2.mp4'     

        pygame.init()
        width, height = 995, 650
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        background = pygame.image.load('images/intersection.png')
        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
        

        #inisialisasi awal
        jalan1 = 6
        jalan2 = '---'
        jalan3 = '---'
        jalan4 = '---'
        lampu_kuning = 5
        signal1 = redSignal
        signal2 = redSignal
        signal3 = redSignal
        signal4 = redSignal


        '''density = 'low'
        condition = 'normal'
        vehicle_counter = 3
        green_light_duration = 5'''

        '''red_light_radius = 20
        red_light_position = (256, 256)  # Koordinat pusat lampu merah'''

        running = True
        perform_detection = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    

            font = pygame.font.Font(None, 30)
            # Gambar garis-garis dan elemen lainnya di sini
            screen.fill((255, 255, 255))
            screen.blit(background,(0,0)) 
            
            
            pygame.draw.rect(screen, (0, 0, 0), (320, 187, 40, 40))
            text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
            screen.blit(text_jalan1, (330, 197))
            
            pygame.draw.rect(screen, (0, 0, 0), (315, 419, 40, 40))
            text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
            screen.blit(text_jalan2, (325, 430))
            
            pygame.draw.rect(screen, (0, 0, 0), (621, 419, 40, 40))
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (630, 430))
            
            pygame.draw.rect(screen, (0, 0, 0), (611, 186, 40, 40))
            text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
            screen.blit(text_jalan4, (620, 197))
            
            
            
            screen.blit(signal1, (360, 138))
            screen.blit(signal2, (355, 418))
            screen.blit(signal3, (590, 418))
            screen.blit(signal4, (580, 138))
            
            

            pygame.display.flip()
            clock.tick(1)

            cek = False
            

            if type(jalan1) == int and jalan1 > 0 and type(jalan4) == str:
                if not cek and type(jalan2) == str:
                    jalan2 = jalan1 + lampu_kuning
                    signal1 = greenSignal
                    cek = True
                
                
                screen.blit(signal1, (360, 138))
                pygame.display.update()
                    
                jalan1 -= 1

                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                
                jalan2 -= 1
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))                           
                    
                if jalan1 == 0 and jalan2 == 5:
                    jalan1 = lampu_kuning +1
                    jalan1 -= 1
                    signal1 = yellowSignal
                    screen.blit(signal1, (360, 138))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))
                    
                if jalan1 == 0 and jalan2 == 0:
                    jalan2 = 10
                    jalan1 = '---'
                    signal1 = redSignal
                    signal2 = greenSignal
                    screen.blit(redSignal, (360, 138))
                    screen.blit(signal2, (355, 418))
                    text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                    screen.blit(text_jalan1, (330, 197))  
                
                    
            elif type(jalan2) == int and jalan2 > 0:
                if not cek and type(jalan3) == str:
                    jalan3 = jalan2 + lampu_kuning
                    cek = True
                      
                jalan2 -= 1
                
                
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (325, 430))
                
                jalan3 -= 1
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
                
                if jalan2 == 0 and jalan3 == 5:
                    jalan2 = lampu_kuning +1
                    jalan2 -= 1
                    signal2 = yellowSignal
                    screen.blit(signal2, (355, 418))
                    text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                    screen.blit(text_jalan2, (325, 430))
                    
                if jalan2 == 0 and jalan3 == 0:
                    jalan3 = 5
                    jalan2 = '---'
                    signal2 = redSignal
                    signal3 = greenSignal
                    screen.blit(signal2, (355, 418))
                    screen.blit(signal3, (590, 418))
                    text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                    screen.blit(text_jalan2, (325, 430))
                
            elif type(jalan3) == int and jalan3 > 0:
                if not cek and type(jalan4) == str:
                    jalan4 = jalan3 + lampu_kuning
                    cek = True
                      
                jalan3 -= 1
                
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (630, 430))
                
                jalan4 -= 1
                text_merah = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_merah, (365, 130))
                
                if jalan3 == 0 and jalan4 == 5:
                    jalan3 = lampu_kuning +1
                    jalan3 -= 1
                    signal3 = yellowSignal
                    screen.blit(signal3, (590, 418))
                    text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                    screen.blit(text_jalan3, (630, 430))
                    
                if jalan3 == 0 and jalan4 == 0:
                    jalan4 = 5
                    jalan3 = '---'
                    screen.blit(redSignal, (590, 418))
                    signal3 = redSignal
                    signal4 = greenSignal
                    text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                    screen.blit(signal4, (580, 138))
                    screen.blit(text_jalan3, (630, 430))
   
            elif type(jalan4) == int and jalan4 > 0:
                if not cek and type(jalan1) == str:
                    jalan1 = jalan4 + lampu_kuning
                    cek = True
                      
                jalan4 -= 1
                               
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (620, 197))
                
                jalan1 -= 1
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (330, 197))
                
                if jalan4 == 0 and jalan1 == 5:
                    jalan4 = lampu_kuning +1
                    jalan4 -= 1
                    signal4 = yellowSignal
                    screen.blit(signal4, (580, 138))
                    text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                    screen.blit(text_jalan4, (620, 197))
                    
                if jalan4 == 0 and jalan1 == 0:
                    jalan1 = 10
                    jalan4 = '---'
                    signal4 = redSignal
                    signal1 = greenSignal
                    screen.blit(signal4, (580, 138))
                    screen.blit(signal1, (360, 138))
                    text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                    screen.blit(text_jalan4, (620, 197))


            
                  
            
            
            '''text_density = font.render(f'Kepadatan: {density}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {condition}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {vehicle_counter}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (330, 60))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {green_light_duration}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (330, 20))'''
            
            
            
            'pygame.draw.circle(screen, (255, 0, 0), red_light_position, red_light_radius)'

            
        pygame.quit()

createJOI()
